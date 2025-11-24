"""
Utility function to convert image file stacks to zarr format.

Scans a directory for image files matching a pattern, groups them into 3D stacks,
and saves each stack as a zarr file with metadata.
"""

from __future__ import annotations

import multiprocessing
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import zarr
except ImportError as err:
    raise ImportError("zarr is required. Install with: pip install zarr") from err


def _load_image(filepath: Path) -> np.ndarray:
    """
    Load an image file using the best available library.

    Parameters
    ----------
    filepath : Path
        Path to image file

    Returns
    -------
    np.ndarray
        Image array, shape (Y, X) or (C, Y, X) or (Y, X, C)
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    # Try tifffile first (best for scientific imaging)
    if tifffile is not None and ext in (".tif", ".tiff"):
        img = tifffile.imread(str(filepath))
        return img

    # Fallback to PIL
    if Image is not None:
        img = Image.open(filepath)
        img_array = np.array(img)
        return img_array

    raise RuntimeError(
        f"Cannot load image {filepath}: No suitable library available. "
        "Install tifffile or Pillow."
    )


def _normalize_axis_order(axis_order: str, has_channels: bool) -> str:
    """
    Normalize and validate axis order.

    Parameters
    ----------
    axis_order : str
        Requested axis order (e.g., "ZCYX")
    has_channels : bool
        Whether image has multiple channels

    Returns
    -------
    str
        Normalized axis order
    """
    axis_order = axis_order.upper()

    if not has_channels:
        # Single channel: always use ZYX
        return "ZYX"

    # Validate axis order contains Z, C, Y, X
    required_axes = {"Z", "C", "Y", "X"}
    if set(axis_order) != required_axes:
        raise ValueError(
            f"axis_order must contain exactly Z, C, Y, X. Got: {axis_order}"
        )

    return axis_order


def _apply_axis_order(
    data: np.ndarray, current_shape: tuple[int, ...], axis_order: str
) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Apply axis order transformation to data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    current_shape : tuple[int, ...]
        Current shape interpretation (Z, C, Y, X) or (Z, Y, X)
    axis_order : str
        Desired axis order (e.g., "ZCYX", "CZYX")

    Returns
    -------
    tuple[np.ndarray, tuple[int, ...]]
        Transformed data and new shape
    """
    if len(current_shape) == 3:
        # Single channel: (Z, Y, X) - no transformation needed
        return data, current_shape

    # Multi-channel: need to reorder
    # Current is always (Z, C, Y, X) from our loading
    # Map to desired order
    current_order = "ZCYX"
    if axis_order == current_order:
        return data, current_shape

    # Create permutation
    perm = [current_order.index(ax) for ax in axis_order]
    data_reordered = np.transpose(data, perm)
    new_shape = tuple(current_shape[i] for i in perm)

    return data_reordered, new_shape


def _load_and_process_image(
    filepath: Path,
    dtype: np.dtype | None,
) -> np.ndarray:
    """
    Load and process a single image file.

    This is a helper function for multiprocessing that loads and processes
    a single image file. It must be a top-level function (not nested) to
    work with multiprocessing.Pool.

    Parameters
    ----------
    filepath : Path
        Path to image file
    dtype : np.dtype | None
        Target dtype for conversion

    Returns
    -------
    np.ndarray
        Processed image array
    """
    img = _load_image(filepath)

    # Normalize to (C, Y, X) if multi-channel
    if img.ndim == 3:
        if img.shape[2] <= 4:  # (Y, X, C)
            img = np.transpose(img, (2, 0, 1))  # (C, Y, X)

    # Convert dtype if needed
    if dtype is not None and img.dtype != dtype:
        img = img.astype(dtype)

    return img


def _load_and_write_to_zarr(
    args: tuple,
) -> tuple[int, bool]:
    """
    Load an image and write it directly to zarr at the specified z-index.

    This function is designed for parallel execution where each worker loads
    and writes a single image, avoiding the need to load all images into memory.
    Zarr supports concurrent writes to different slices, so this enables true parallelism.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - z_idx: int - Z-index in the zarr array
        - filepath: Path - Path to image file
        - zarr_path: str - Path to zarr array
        - final_shape: tuple - Final shape of zarr array
        - dtype: np.dtype | None - Target dtype
        - has_channels: bool - Whether image has channels
        - axis_order: str - Final axis order (e.g., "ZCYX", "CZYX")
        - C: int - Number of channels (for multi-channel)
        - Y: int - Image height
        - X: int - Image width

    Returns
    -------
    tuple[int, bool]
        (z_idx, success) tuple indicating which z-index was written
    """
    (
        z_idx,
        filepath,
        zarr_path,
        final_shape,
        dtype,
        has_channels,
        axis_order,
        C,
        Y,
        X,
    ) = args

    try:
        # Load and process image
        img = _load_and_process_image(filepath, dtype)

        # Open zarr array (read-write mode supports concurrent writes)
        zarr_array = zarr.open(zarr_path, mode="r+")

        if has_channels:
            # Apply axis order transformation for this slice
            # We have img as (C, Y, X), need to write at z_idx
            if axis_order == "CZYX":
                # Write to (C, Z, Y, X) array
                zarr_array[:, z_idx, :, :] = img
            elif axis_order == "ZCYX":
                # Write to (Z, C, Y, X) array
                zarr_array[z_idx, :, :, :] = img
            else:
                # For other axis orders, we need to reorder the slice
                # Create a (1, C, Y, X) array, apply transformation, then write
                slice_data = img[np.newaxis, ...]  # (1, C, Y, X)
                slice_reordered, _ = _apply_axis_order(
                    slice_data, (1, C, Y, X), axis_order
                )
                # Write based on first dimension position
                if axis_order[0] == "Z":
                    # Z is first: write to z_idx position
                    zarr_array[z_idx, ...] = slice_reordered[0]
                else:
                    # C is first: write to z_idx on second dim
                    zarr_array[:, z_idx, ...] = slice_reordered[:, 0, ...]
        else:
            # Single channel: write directly to (Z, Y, X) array
            zarr_array[z_idx, :, :] = img

        return (z_idx, True)
    except Exception as e:
        print(f"Error processing {filepath} at z_idx {z_idx}: {e}")
        import traceback

        traceback.print_exc()
        return (z_idx, False)


def stack_files_to_zarr(
    directory: str | Path,
    extension: str,
    pattern: str | re.Pattern,
    output_dir: str | Path | None = None,
    zarr_chunks: tuple[int, ...] | None = None,
    dtype: np.dtype | None = None,
    axis_order: str = "ZCYX",
    output_naming: Callable[[str], str] | None = None,
    sort_by_counter: bool = True,
    dry_run: bool = False,
    num_workers: int | None = None,
) -> dict[str, dict]:
    """
    Scan directory for image files, group into 3D stacks, and save as zarr.

    Parameters
    ----------
    directory : str | Path
        Directory to scan for image files (top level only, non-recursive)
    extension : str
        File extension to match (e.g., '.tif', '.png')
    pattern : str | re.Pattern
        Regex pattern with two groups: (basename, counter)
        Example: r"(.+)_(\\d+)\\.tif$"
    output_dir : str | Path | None
        Directory to save zarr files. If None, saves in same directory.
    zarr_chunks : tuple[int, ...] | None
        Chunk size for zarr arrays. If None, uses reasonable defaults.
    dtype : np.dtype | None
        Data type for zarr arrays. If None, infers from first image.
    axis_order : str
        Axis order for multi-channel images. Default: "ZCYX"
        Options: "ZCYX", "CZYX", "ZYCX", etc.
        Single channel images always use "ZYX" regardless of this setting.
    output_naming : Callable[[str], str] | None
        Function to generate output zarr filename from basename.
        If None, uses default: f"{basename}.zarr"
        Example: lambda b: f"{b}_stack.zarr"
    sort_by_counter : bool
        Whether to sort files by counter value (default: True)
    dry_run : bool
        If True, only analyze files without creating zarr (default: False)
    num_workers : int | None
        Number of worker processes for parallel image loading. If None, uses
        number of CPU cores. If 0 or 1, disables multiprocessing (default: None)

    Returns
    -------
    dict[str, dict]
        Dictionary mapping stack basename to metadata:
        {
            "stack_name": {
                "zarr_path": str,
                "shape": tuple[int, ...],  # (Z, C, Y, X) or (Z, Y, X)
                "dtype": np.dtype,
                "file_count": int,
                "files": list[str],  # Sorted list of file paths
                "counter_range": tuple[int, int],  # (min, max)
                "axis_order": str,  # Actual axis order used
            }
        }

    Examples
    --------
    >>> from qlty.utils.stack_to_zarr import stack_files_to_zarr
    >>> result = stack_files_to_zarr(
    ...     directory="/path/to/images",
    ...     extension=".tif",
    ...     pattern=r"(.+)_(\\d+)\\.tif$",
    ...     output_dir="/path/to/zarr_output"
    ... )
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    # Normalize extension
    if not extension.startswith("."):
        extension = "." + extension
    extension = extension.lower()

    # Compile pattern
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    # Step 1: File Discovery and Parsing
    stacks: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    for filepath in directory.iterdir():
        if not filepath.is_file():
            continue

        # Check extension
        if filepath.suffix.lower() != extension:
            continue

        # Match pattern
        match = pattern.match(filepath.name)
        if not match:
            continue

        if match.lastindex is None or match.lastindex < 1:
            raise ValueError(
                "Pattern must have at least 2 groups (basename, counter). "
                "Pattern has no groups."
            )

        if match.lastindex < 2:
            raise ValueError(
                f"Pattern must have at least 2 groups (basename, counter). "
                f"Got {match.lastindex} groups."
            )

        basename = match.group(1)
        counter_str = match.group(2)

        try:
            counter = int(counter_str)
        except ValueError:
            continue  # Skip if counter not parseable

        stacks[basename].append((counter, filepath))

    if not stacks:
        return {}

    # Step 2: Stack Analysis
    results = {}

    for basename, file_list in stacks.items():
        # Sort by counter
        if sort_by_counter:
            file_list.sort(key=lambda x: x[0])

        counters = [c for c, _ in file_list]
        counter_min = min(counters)
        counter_max = max(counters)

        # Check for gaps
        expected_counters = set(range(counter_min, counter_max + 1))
        actual_counters = set(counters)
        missing = expected_counters - actual_counters
        if missing:
            print(
                f"Warning: Stack '{basename}' has missing counters: {sorted(missing)}"
            )

        # Load first image to determine dimensions
        first_file = file_list[0][1]
        first_image = _load_image(first_file)

        # Determine shape
        if first_image.ndim == 2:
            # Single channel: (Y, X)
            Y, X = first_image.shape
            C = 1
            has_channels = False
            final_axis_order = "ZYX"
            final_shape = (len(file_list), Y, X)
        elif first_image.ndim == 3:
            # Multi-channel: could be (C, Y, X) or (Y, X, C)
            # Assume (Y, X, C) if last dim is small, else (C, Y, X)
            if first_image.shape[2] <= 4:  # Likely (Y, X, C)
                Y, X, C = first_image.shape
                first_image = np.transpose(first_image, (2, 0, 1))  # (C, Y, X)
            else:  # Likely (C, Y, X)
                C, Y, X = first_image.shape
            has_channels = True
            final_axis_order = _normalize_axis_order(axis_order, has_channels)
            # Start with ZCYX, will apply axis_order later
            base_shape = (len(file_list), C, Y, X)
            _, final_shape = _apply_axis_order(
                np.zeros(base_shape, dtype=first_image.dtype),
                base_shape,
                final_axis_order,
            )
        else:
            raise ValueError(
                f"Unsupported image dimensions: {first_image.ndim}D. "
                "Expected 2D (Y, X) or 3D (C, Y, X) or (Y, X, C)."
            )

        # Determine dtype
        if dtype is None:
            dtype = first_image.dtype
        else:
            dtype = np.dtype(dtype)

        # Validate all images have same dimensions
        for _counter, filepath in file_list[1:]:
            img = _load_image(filepath)
            if img.ndim == 2:
                if img.shape != (Y, X):
                    raise ValueError(
                        f"Image {filepath} has shape {img.shape}, expected ({Y}, {X})"
                    )
            elif img.ndim == 3:
                if img.shape[2] <= 4:
                    # (Y, X, C) format
                    img_Y, img_X, img_C = img.shape
                    if img.shape[:2] != (Y, X) or img_C != C:
                        raise ValueError(
                            f"Image {filepath} has shape {img.shape}, "
                            f"expected ({Y}, {X}, {C})"
                        )
                else:
                    # (C, Y, X) format
                    img_C, img_Y, img_X = img.shape
                    if img.shape[1:] != (Y, X) or img_C != C:
                        raise ValueError(
                            f"Image {filepath} has shape {img.shape}, "
                            f"expected ({C}, {Y}, {X})"
                        )

        # Determine output path
        if output_naming is not None:
            zarr_name = output_naming(basename)
        else:
            zarr_name = f"{basename}.zarr"

        if output_dir is not None:
            output_path = Path(output_dir) / zarr_name
        else:
            output_path = directory / zarr_name

        # Determine chunk size
        if zarr_chunks is None:
            if has_channels:
                # Default: one z-slice per chunk
                if final_axis_order == "ZCYX":
                    zarr_chunks = (1, C, Y, X)
                elif final_axis_order == "CZYX":
                    zarr_chunks = (C, 1, Y, X)
                else:
                    # Generic: use first dimension as 1
                    zarr_chunks = (1,) + final_shape[1:]
            else:
                zarr_chunks = (1, Y, X)

        # Step 3: Zarr Creation (if not dry run)
        if not dry_run:
            # Create zarr array
            zarr_array = zarr.open(
                str(output_path),
                mode="w",
                shape=final_shape,
                chunks=zarr_chunks,
                dtype=dtype,
            )

            # Determine if we should use multiprocessing
            use_multiprocessing = False
            if num_workers is None:
                # Auto-detect: use multiprocessing if more than 1 CPU core
                use_multiprocessing = multiprocessing.cpu_count() > 1
                workers = multiprocessing.cpu_count()
            elif num_workers > 1:
                use_multiprocessing = True
                workers = num_workers
            else:
                workers = 1

            # For large stacks, use parallel load-and-write to avoid loading all into memory
            # and to enable parallel zarr writes
            if use_multiprocessing and len(file_list) > 10:  # Use for large stacks
                # Parallel load-and-write: each worker loads an image and writes it directly
                # This avoids loading all images into memory and enables parallel zarr writes
                # Zarr supports concurrent writes to different slices
                tasks = []
                for z_idx, (_, filepath) in enumerate(file_list):
                    tasks.append(
                        (
                            z_idx,
                            filepath,
                            str(output_path),
                            final_shape,
                            dtype,
                            has_channels,
                            final_axis_order,
                            C,
                            Y,
                            X,
                        )
                    )

                # Use all available workers for maximum parallelism
                with multiprocessing.Pool(processes=workers) as pool:
                    # Process in parallel - each worker loads and writes one image
                    # Using imap_unordered for better performance with many tasks
                    write_results = list(pool.imap_unordered(_load_and_write_to_zarr, tasks))
                    # Check for failures
                    failures = [r for r in write_results if not r[1]]
                    if failures:
                        print(
                            f"Warning: {len(failures)} images failed to write out of {len(file_list)}"
                        )
            else:
                # Sequential or small stack: load all first, then write
                if use_multiprocessing and len(file_list) > 1:
                    # Parallel loading only
                    load_func = partial(_load_and_process_image, dtype=dtype)
                    with multiprocessing.Pool(processes=workers) as pool:
                        filepaths = [f for _, f in file_list]
                        images = pool.map(load_func, filepaths)
                else:
                    # Sequential loading
                    images = [
                        _load_and_process_image(filepath, dtype=dtype)
                        for _, filepath in file_list
                    ]

                # Write images to zarr
                if has_channels:
                    # Need to apply axis order
                    # We have (C, Y, X), need to stack as (Z, C, Y, X) then reorder
                    stack_data = np.zeros((len(file_list), C, Y, X), dtype=dtype)
                    for z_idx, img in enumerate(images):
                        stack_data[z_idx] = img

                    # Apply axis order and write
                    stack_reordered, _ = _apply_axis_order(
                        stack_data, (len(file_list), C, Y, X), final_axis_order
                    )
                    zarr_array[:] = stack_reordered
                else:
                    # Single channel: direct write
                    for z_idx, img in enumerate(images):
                        zarr_array[z_idx] = img

            # Store metadata as zarr attributes
            zarr_array.attrs.update(
                {
                    "basename": basename,
                    "file_count": len(file_list),
                    "counter_range": [counter_min, counter_max],
                    "axis_order": final_axis_order,
                    "files": [str(f) for _, f in file_list],
                    "pattern": pattern.pattern
                    if isinstance(pattern, re.Pattern)
                    else pattern,
                    "extension": extension,
                }
            )

        # Store results
        results[basename] = {
            "zarr_path": str(output_path),
            "shape": final_shape,
            "dtype": dtype,
            "file_count": len(file_list),
            "files": [str(f) for _, f in file_list],
            "counter_range": (counter_min, counter_max),
            "axis_order": final_axis_order,
        }

    return results
