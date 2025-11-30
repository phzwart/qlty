"""
Utility functions to convert image file stacks to zarr and OME-Zarr formats.

Scans a directory for image files matching a pattern, groups them into 3D stacks,
and saves each stack as:
- Standard zarr files with metadata (stack_files_to_zarr)
- OME-Zarr format with multiscale pyramids (stack_files_to_ome_zarr)

OME-Zarr follows the Next-Generation File Format (NGFF) specification for bioimaging data
and supports image pyramids for efficient multi-resolution access.
"""

from __future__ import annotations

import multiprocessing
import re
import time
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
    msg = "zarr is required. Install with: pip install zarr"
    raise ImportError(msg) from err

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _create_zarr_array(group, name, **kwargs):
    """
    Create a zarr array in a group (requires zarr >= 3.0.0a5).

    Parameters
    ----------
    group : zarr.Group
        The zarr group to create the array in
    name : str
        Name of the array
    **kwargs
        Additional arguments passed to create()
        If 'data' is provided, shape and dtype will be extracted from it

    Returns
    -------
    zarr.Array
        The created zarr array
    """
    # Extract data if provided
    data = kwargs.pop("data", None)

    # If data is provided, extract shape and dtype for zarr 3.0.0a5 compatibility
    # zarr 3.0.0a5's create() calls create_array() which requires shape as keyword-only arg
    if data is not None:
        kwargs["shape"] = data.shape
        kwargs["dtype"] = data.dtype
        arr = group.create(name, **kwargs)
        arr[:] = data
        return arr

    return group.create(name, **kwargs)


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
        return tifffile.imread(str(filepath))

    # Fallback to PIL
    if Image is not None:
        img = Image.open(filepath)
        return np.array(img)

    msg = (
        f"Cannot load image {filepath}: No suitable library available. "
        "Install tifffile or Pillow."
    )
    raise RuntimeError(
        msg,
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
        msg = f"axis_order must contain exactly Z, C, Y, X. Got: {axis_order}"
        raise ValueError(
            msg,
        )

    return axis_order


def _apply_axis_order(
    data: np.ndarray,
    current_shape: tuple[int, ...],
    axis_order: str,
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
    if img.ndim == 3 and img.shape[2] <= 4:  # (Y, X, C)
        img = np.transpose(img, (2, 0, 1))  # (C, Y, X)

    # Convert dtype if needed
    if dtype is not None and img.dtype != dtype:
        img = img.astype(dtype)

    return img


def _load_and_write_to_ome_zarr_base(
    args: tuple,
) -> tuple[int, bool]:
    """
    Load an image and write it directly to OME-Zarr base level at the specified z-index.
    
    This function is designed for parallel execution where each worker loads
    and writes a single image, avoiding the need to load all images into memory.
    Zarr supports concurrent writes to different slices, so this enables true parallelism.
    
    Parameters
    ----------
    args : tuple
        Tuple containing:
        - z_idx: int - Z-index in the zarr array
        - filepath: Path - Path to image file
        - zarr_group_path: str - Path to zarr group (OME-Zarr root)
        - array_name: str - Name of array in group (e.g., "0" for base level)
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
        zarr_group_path,
        array_name,
        _final_shape,
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
        
        # Open zarr group and array (read-write mode supports concurrent writes)
        zarr_group = zarr.open_group(zarr_group_path, mode="r+")
        zarr_array = zarr_group[array_name]
        
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
                    slice_data,
                    (1, C, Y, X),
                    axis_order,
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
    except Exception:
        import traceback
        traceback.print_exc()
        return (z_idx, False)


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
        _final_shape,
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
                    slice_data,
                    (1, C, Y, X),
                    axis_order,
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
    except Exception:
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
        msg = f"Directory does not exist: {directory}"
        raise ValueError(msg)

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
            msg = (
                "Pattern must have at least 2 groups (basename, counter). "
                "Pattern has no groups."
            )
            raise ValueError(
                msg,
            )

        if match.lastindex < 2:
            msg = (
                f"Pattern must have at least 2 groups (basename, counter). "
                f"Got {match.lastindex} groups."
            )
            raise ValueError(
                msg,
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

    for _stack_idx, (basename, file_list) in enumerate(stacks.items(), 1):
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
                f"Warning: Stack '{basename}' has missing counters: {sorted(missing)}",
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
            msg = (
                f"Unsupported image dimensions: {first_image.ndim}D. "
                "Expected 2D (Y, X) or 3D (C, Y, X) or (Y, X, C)."
            )
            raise ValueError(
                msg,
            )

        # Determine dtype
        dtype = first_image.dtype if dtype is None else np.dtype(dtype)

        # Validate all images have same dimensions
        for _counter, filepath in file_list[1:]:
            img = _load_image(filepath)
            if img.ndim == 2:
                if img.shape != (Y, X):
                    msg = f"Image {filepath} has shape {img.shape}, expected ({Y}, {X})"
                    raise ValueError(
                        msg,
                    )
            elif img.ndim == 3:
                if img.shape[2] <= 4:
                    # (Y, X, C) format
                    img_Y, img_X, img_C = img.shape
                    if img.shape[:2] != (Y, X) or img_C != C:
                        msg = (
                            f"Image {filepath} has shape {img.shape}, "
                            f"expected ({Y}, {X}, {C})"
                        )
                        raise ValueError(
                            msg,
                        )
                else:
                    # (C, Y, X) format
                    img_C, _img_Y, _img_X = img.shape
                    if img.shape[1:] != (Y, X) or img_C != C:
                        msg = (
                            f"Image {filepath} has shape {img.shape}, "
                            f"expected ({C}, {Y}, {X})"
                        )
                        raise ValueError(
                            msg,
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
                    zarr_chunks = (1, *final_shape[1:])
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

            if use_multiprocessing:
                pass

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
                        ),
                    )

                # Use all available workers for maximum parallelism
                with multiprocessing.Pool(processes=workers) as pool:
                    # Process in parallel - each worker loads and writes one image
                    # Using imap_unordered for better performance with many tasks
                    if tqdm is not None:
                        write_results = list(
                            tqdm(
                                pool.imap_unordered(_load_and_write_to_zarr, tasks),
                                total=len(tasks),
                                desc=f"  Processing {basename}",
                                unit="image",
                            ),
                        )
                    else:
                        # Fallback: process with periodic status updates
                        write_results = []
                        completed = 0
                        for result in pool.imap_unordered(
                            _load_and_write_to_zarr,
                            tasks,
                        ):
                            write_results.append(result)
                            completed += 1
                            if completed % max(
                                1,
                                len(tasks) // 20,
                            ) == 0 or completed == len(tasks):
                                pass

                    # Check for failures
                    failures = [r for r in write_results if not r[1]]
                    if failures:
                        pass
            else:
                # Sequential or small stack: load all first, then write
                if use_multiprocessing and len(file_list) > 1:
                    # Parallel loading only
                    load_func = partial(_load_and_process_image, dtype=dtype)
                    with multiprocessing.Pool(processes=workers) as pool:
                        filepaths = [f for _, f in file_list]
                        if tqdm is not None:
                            images = list(
                                tqdm(
                                    pool.imap(load_func, filepaths),
                                    total=len(filepaths),
                                    desc="  Loading images",
                                    unit="image",
                                ),
                            )
                        else:
                            images = pool.map(load_func, filepaths)
                # Sequential loading
                elif tqdm is not None:
                    images = [
                        _load_and_process_image(filepath, dtype=dtype)
                        for filepath in tqdm(
                            [f for _, f in file_list],
                            desc="  Loading images",
                            unit="image",
                        )
                    ]
                else:
                    images = []
                    for idx, (_, filepath) in enumerate(file_list, 1):
                        images.append(
                            _load_and_process_image(filepath, dtype=dtype),
                        )
                        if idx % max(1, len(file_list) // 20) == 0 or idx == len(
                            file_list,
                        ):
                            pass

                # Write images to zarr
                if has_channels:
                    # Need to apply axis order
                    # We have (C, Y, X), need to stack as (Z, C, Y, X) then reorder
                    stack_data = np.zeros((len(file_list), C, Y, X), dtype=dtype)
                    for z_idx, img in enumerate(images):
                        stack_data[z_idx] = img

                    # Apply axis order and write
                    stack_reordered, _ = _apply_axis_order(
                        stack_data,
                        (len(file_list), C, Y, X),
                        final_axis_order,
                    )
                    zarr_array[:] = stack_reordered
                # Single channel: direct write
                elif tqdm is not None:
                    for z_idx, img in enumerate(
                        tqdm(images, desc="  Writing to zarr", unit="image"),
                    ):
                        zarr_array[z_idx] = img
                else:
                    for z_idx, img in enumerate(images):
                        zarr_array[z_idx] = img
                        if (z_idx + 1) % max(1, len(images) // 20) == 0 or (
                            z_idx + 1
                        ) == len(images):
                            pass

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
                },
            )
        else:
            pass

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


def stack_files_to_ome_zarr(
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
    pyramid_levels: int | None = None,
    pyramid_scale_factors: list[tuple[int, ...]] | None = None,
    downsample_mode: str = "2d",
    downsample_axes: tuple[str, ...] | None = None,
    downsample_method: str = "dask_coarsen",
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Scan directory for image files, group into 3D stacks, and save as OME-Zarr with pyramids.

    Creates OME-Zarr format files with multiscale image pyramids (multiple resolution levels).
    OME-Zarr follows the Next-Generation File Format (NGFF) specification for bioimaging data.

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
        Directory to save OME-Zarr files. If None, saves in same directory.
    zarr_chunks : tuple[int, ...] | None
        Chunk size for base resolution zarr arrays. If None, uses reasonable defaults.
    dtype : np.dtype | None
        Data type for zarr arrays. If None, infers from first image.
    axis_order : str
        Axis order for multi-channel images. Default: "ZCYX"
        Options: "ZCYX", "CZYX", "ZYCX", etc.
        Single channel images always use "ZYX" regardless of this setting.
        Note: OME-Zarr standard uses "TCZYX" but we use "ZCYX" for compatibility.
    output_naming : Callable[[str], str] | None
        Function to generate output zarr filename from basename.
        If None, uses default: f"{basename}.ome.zarr"
        Example: lambda b: f"{b}_stack.ome.zarr"
    sort_by_counter : bool
        Whether to sort files by counter value (default: True)
    dry_run : bool
        If True, only analyze files without creating zarr (default: False)
    num_workers : int | None
        Number of worker processes for parallel image loading. If None, uses
        number of CPU cores. If 0 or 1, disables multiprocessing (default: None)
    pyramid_levels : int | None
        Number of pyramid levels to create (including base level).
        If None, automatically determines based on image size.
        Example: pyramid_levels=4 creates 4 resolution levels (1x, 2x, 4x, 8x downsampled).
    pyramid_scale_factors : list[tuple[int, ...]] | None
        Custom scale factors for each pyramid level (excluding base level).
        Each tuple specifies scale factors for each dimension (Z, C, Y, X).
        If None, uses automatic 2x downsampling per level.
        Example: [(1, 1, 2, 2), (1, 1, 4, 4)] creates 2 pyramid levels with 2x and 4x downsampling in Y/X.
    downsample_mode : str
        Downsampling mode for pyramid generation. Default: "2d"
        - "2d": For 2D operations on 3D grid - downsample only Y, X (not Z)
        - "3d": For pure 3D work - downsample Z, Y, X
        Ignored if downsample_axes is provided.
    downsample_axes : tuple[str, ...] | None
        Explicit control over which axes to downsample. If None, auto-determined from downsample_mode.
        Options: ("z", "y", "x") or ("y", "x") or ("y",) or ("x",)
        Takes precedence over downsample_mode.
    downsample_method : str
        Downsampling algorithm to use. Default: "dask_coarsen"
        - "dask_coarsen": Use Dask coarsen (fast, parallel, recommended)
        - "scipy_zoom": Use scipy.ndimage.zoom (fallback)
        Future methods can be added (e.g., "block_average")
    verbose : bool
        Whether to print detailed progress information. Default: True
        When True, prints:
        - File discovery and stack information
        - Image dimensions and memory estimates
        - Loading progress
        - Pyramid level creation progress with timing
        - Dask configuration details

    Returns
    -------
    dict[str, dict]
        Dictionary mapping stack basename to metadata:
        {
            "stack_name": {
                "zarr_path": str,
                "shape": tuple[int, ...],  # (Z, C, Y, X) or (Z, Y, X) - base level
                "dtype": np.dtype,
                "file_count": int,
                "files": list[str],  # Sorted list of file paths
                "counter_range": tuple[int, int],  # (min, max)
                "axis_order": str,  # Actual axis order used
                "pyramid_levels": int,  # Number of pyramid levels created
            }
        }

    Examples
    --------
    >>> from qlty.utils.stack_to_zarr import stack_files_to_ome_zarr
    >>> result = stack_files_to_ome_zarr(
    ...     directory="/path/to/images",
    ...     extension=".tif",
    ...     pattern=r"(.+)_(\\d+)\\.tif$",
    ...     output_dir="/path/to/ome_zarr_output",
    ...     pyramid_levels=4  # Create 4 resolution levels
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

    # Reuse file discovery logic from stack_files_to_zarr
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
        if verbose:
            print("No matching files found.")
        return {}

    if verbose:
        print(f"Found {len(stacks)} stack(s) to process")
        print(f"Scanning directory: {directory}")
        print(f"File pattern: {pattern.pattern if isinstance(pattern, re.Pattern) else pattern}")
        print(f"Extension: {extension}")

    # Step 2: Stack Analysis (reuse logic from stack_files_to_zarr)
    results = {}

    for stack_idx, (basename, file_list) in enumerate(stacks.items(), 1):
        if verbose:
            print(
                f"\n{'='*70}"
            )
            print(
                f"[{stack_idx}/{len(stacks)}] Processing stack: {basename}"
            )
            print(f"  Files found: {len(file_list)}")
            print(f"  Counter range: {min(c for c, _ in file_list)} - {max(c for c, _ in file_list)}")
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
            base_shape = (len(file_list), Y, X)
        elif first_image.ndim == 3:
            # Multi-channel: could be (C, Y, X) or (Y, X, C)
            if first_image.shape[2] <= 4:  # Likely (Y, X, C)
                Y, X, C = first_image.shape
                first_image = np.transpose(first_image, (2, 0, 1))  # (C, Y, X)
            else:  # Likely (C, Y, X)
                C, Y, X = first_image.shape
            has_channels = True
            final_axis_order = _normalize_axis_order(axis_order, has_channels)
            # Start with ZCYX, will apply axis_order later
            base_shape_ordered = (len(file_list), C, Y, X)
            _, final_shape_tuple = _apply_axis_order(
                np.zeros(base_shape_ordered, dtype=first_image.dtype),
                base_shape_ordered,
                final_axis_order,
            )
            base_shape = final_shape_tuple
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
        
        if verbose:
            print(f"  Image dimensions: {first_image.shape}")
            print(f"  Detected shape: {base_shape}")
            print(f"  Data type: {dtype}")
            print(f"  Axis order: {final_axis_order}")
            if has_channels:
                print(f"  Channels: {C}")
            # Calculate approximate memory size
            import sys
            element_size = np.dtype(dtype).itemsize
            total_elements = np.prod(base_shape)
            memory_gb = (total_elements * element_size) / (1024**3)
            print(f"  Estimated memory per level: {memory_gb:.2f} GB")

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
                    img_Y, img_X, img_C = img.shape
                    if img.shape[:2] != (Y, X) or img_C != C:
                        raise ValueError(
                            f"Image {filepath} has shape {img.shape}, "
                            f"expected ({Y}, {X}, {C})"
                        )
                else:
                    img_C, img_Y, img_X = img.shape
                    if img.shape[1:] != (Y, X) or img_C != C:
                        raise ValueError(
                            f"Image {filepath} has shape {img.shape}, "
                            f"expected ({C}, {Y}, {X})"
                        )

        # Determine output path
        if output_naming is not None:
            zarr_name = output_naming(basename)
            if not zarr_name.endswith(".ome.zarr"):
                zarr_name = zarr_name.replace(".zarr", ".ome.zarr")
                if not zarr_name.endswith(".ome.zarr"):
                    zarr_name = f"{zarr_name}.ome.zarr"
        else:
            zarr_name = f"{basename}.ome.zarr"

        if output_dir is not None:
            output_path = Path(output_dir) / zarr_name
        else:
            output_path = directory / zarr_name

        # Determine pyramid levels and scale factors
        if pyramid_scale_factors is not None:
            num_pyramid_levels = len(pyramid_scale_factors) + 1  # +1 for base level
        elif pyramid_levels is not None:
            num_pyramid_levels = pyramid_levels
        else:
            # Auto-determine: create pyramid until smallest dimension is < 256
            min_dim = min(Y, X)
            num_pyramid_levels = 1
            dim = min_dim
            while dim > 256:
                dim = dim // 2
                num_pyramid_levels += 1
            num_pyramid_levels = max(1, min(num_pyramid_levels, 5))  # Limit to 5 levels

        # Determine which axes to downsample
        if downsample_axes is not None:
            axes_to_downsample = set(downsample_axes)
        elif downsample_mode == "2d":
            # 2D mode: don't downsample Z, only Y and X
            axes_to_downsample = {"y", "x"}
        elif downsample_mode == "3d":
            # 3D mode: downsample Z, Y, X
            axes_to_downsample = {"z", "y", "x"}
        else:
            raise ValueError(
                f"Invalid downsample_mode: {downsample_mode}. Must be '2d' or '3d'."
            )

        # Generate scale factors if not provided
        if pyramid_scale_factors is None:
            pyramid_scale_factors = []
            for level in range(1, num_pyramid_levels):
                scale = 2**level
                # OME-Zarr format: scale factors are per dimension (Z, C, Y, X)
                if has_channels:
                    if final_axis_order == "ZCYX":
                        z_scale = scale if "z" in axes_to_downsample else 1
                        c_scale = 1  # Never downsample channels
                        y_scale = scale if "y" in axes_to_downsample else 1
                        x_scale = scale if "x" in axes_to_downsample else 1
                        pyramid_scale_factors.append(
                            (z_scale, c_scale, y_scale, x_scale)
                        )
                    elif final_axis_order == "CZYX":
                        c_scale = 1  # Never downsample channels
                        z_scale = scale if "z" in axes_to_downsample else 1
                        y_scale = scale if "y" in axes_to_downsample else 1
                        x_scale = scale if "x" in axes_to_downsample else 1
                        pyramid_scale_factors.append(
                            (c_scale, z_scale, y_scale, x_scale)
                        )
                    else:
                        # Generic: don't scale C, scale others based on axes_to_downsample
                        z_scale = scale if "z" in axes_to_downsample else 1
                        y_scale = scale if "y" in axes_to_downsample else 1
                        x_scale = scale if "x" in axes_to_downsample else 1
                        pyramid_scale_factors.append(
                            (1, 1, y_scale, x_scale)
                        )  # Default ZCYX order
                else:
                    # Single channel: (Z, Y, X)
                    z_scale = scale if "z" in axes_to_downsample else 1
                    y_scale = scale if "y" in axes_to_downsample else 1
                    x_scale = scale if "x" in axes_to_downsample else 1
                    pyramid_scale_factors.append((z_scale, y_scale, x_scale))

        if not dry_run:
            if verbose:
                print(f"  Creating OME-Zarr: {output_path}", flush=True)
                print(f"  Base shape: {base_shape}, dtype: {dtype}", flush=True)
                print(f"  Pyramid levels: {num_pyramid_levels}", flush=True)
                print(f"  Downsample method: {downsample_method}", flush=True)
                print(f"  Downsample mode: {downsample_mode}", flush=True)
                print(f"\n  *** STARTING PROCESSING - THIS MAY TAKE A WHILE ***", flush=True)
                print(f"  *** WATCH FOR PROGRESS BARS BELOW ***\n", flush=True)

            # Create OME-Zarr root group
            if verbose:
                print("  Creating zarr root group...", flush=True)
            root = zarr.open_group(str(output_path), mode="w")
            multiscales_metadata = []
            if verbose:
                print("  ✓ Zarr root group created", flush=True)

            # Store base level data
            # OPTIMIZATION: Write directly to zarr in parallel instead of loading all into memory
            # This is MUCH faster and uses way less memory (like stack_files_to_zarr)
            
            # Determine chunk size for base level
            if zarr_chunks is None:
                if has_channels:
                    if final_axis_order == "ZCYX":
                        base_chunks = (1, min(C, 4), min(Y, 256), min(X, 256))
                    elif final_axis_order == "CZYX":
                        base_chunks = (min(C, 4), 1, min(Y, 256), min(X, 256))
                    else:
                        base_chunks = (1,) + tuple(min(d, 256) for d in base_shape[1:])
                else:
                    base_chunks = (1, min(Y, 256), min(X, 256))
            else:
                base_chunks = zarr_chunks

            # Create base level array FIRST (empty, we'll write to it in parallel)
            if verbose:
                print("\n    Creating base level zarr array (empty, will write in parallel)...", flush=True)
            base_zarr_array = root.create(
                "0",
                shape=base_shape,
                chunks=base_chunks,
                dtype=dtype,
            )
            if verbose:
                print("    ✓ Base zarr array created", flush=True)
                print("\n" + "="*70, flush=True)
                print("  [STEP 1/3] WRITING IMAGES DIRECTLY TO ZARR - FAST PARALLEL MODE", flush=True)
                print("="*70 + "\n", flush=True)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
            # Load images (reuse logic from stack_files_to_zarr)
            # Use same worker count as Dask will use for consistency
            import multiprocessing
            if num_workers is None:
                num_cores = multiprocessing.cpu_count()
                use_multiprocessing = num_cores > 1
                workers = num_cores
            elif num_workers > 1:
                use_multiprocessing = True
                workers = num_workers
                num_cores = num_workers
            else:
                workers = 1
                num_cores = 1
                use_multiprocessing = False
            
            if verbose:
                if use_multiprocessing:
                    print(f"    Using multiprocessing with {workers} workers (one per core)", flush=True)
                    print(f"    Writing {len(file_list)} images directly to zarr in parallel...", flush=True)
                else:
                    print(f"    Using sequential writing (1 worker)", flush=True)
                    print(f"    Writing {len(file_list)} images...", flush=True)

            # OPTIMIZED: Write directly to zarr in parallel (like stack_files_to_zarr)
            # This avoids loading all images into memory - MUCH faster!
            if use_multiprocessing and len(file_list) > 10:
                # Prepare tasks for parallel load-and-write
                tasks = []
                for z_idx, (_, filepath) in enumerate(file_list):
                    tasks.append(
                        (
                            z_idx,
                            filepath,
                            str(output_path),  # zarr group path
                            "0",  # array name (base level)
                            base_shape,
                            dtype,
                            has_channels,
                            final_axis_order,
                            C,
                            Y,
                            X,
                        ),
                    )
                if verbose:
                    print(f"\n    Starting multiprocessing pool with {workers} workers...", flush=True)
                    print(f"    Images will be written directly to zarr across {workers} cores", flush=True)
                    # Verify actual worker count
                    try:
                        import psutil
                        actual_cpu_count = psutil.cpu_count(logical=False)  # Physical cores
                        logical_cpu_count = psutil.cpu_count(logical=True)  # Logical cores
                        print(f"    DEBUG: System has {actual_cpu_count} physical cores, {logical_cpu_count} logical cores", flush=True)
                        print(f"    DEBUG: Requested workers = {workers}", flush=True)
                    except ImportError:
                        print(f"    DEBUG: multiprocessing.cpu_count() = {multiprocessing.cpu_count()}", flush=True)
                        print(f"    DEBUG: Requested workers = {workers}", flush=True)
                    print(f"    WRITING {len(file_list)} IMAGES DIRECTLY TO ZARR - PROGRESS BAR BELOW:", flush=True)
                    print("-"*70, flush=True)
                
                # Create pool and verify it actually created workers
                pool = multiprocessing.Pool(processes=workers)
                if verbose:
                    try:
                        import psutil
                        current_process = psutil.Process()
                        children = current_process.children(recursive=True)
                        print(f"    DEBUG: Pool created, active child processes: {len(children)}", flush=True)
                        if len(children) < workers:
                            print(f"    WARNING: Only {len(children)} child processes created, expected {workers}!", flush=True)
                    except ImportError:
                        pass
                
                try:
                    # Write directly to zarr in parallel (like stack_files_to_zarr)
                    # Using imap_unordered for better performance with many tasks
                    chunksize = max(1, len(tasks) // (workers * 4))
                    if verbose:
                        print(f"    DEBUG: Using chunksize={chunksize} for better load balancing", flush=True)
                    
                    if tqdm is not None:
                        if verbose:
                            print("", flush=True)  # Blank line before progress bar
                        write_results = list(
                            tqdm(
                                pool.imap_unordered(_load_and_write_to_ome_zarr_base, tasks, chunksize=chunksize),
                                total=len(tasks),
                                desc="    WRITING TO ZARR",
                                unit="img",
                                ncols=100,
                                miniters=1,
                            )
                        )
                        if verbose:
                            print("", flush=True)  # Blank line after progress bar
                    else:
                        # Manual progress bar when tqdm not available
                        if verbose:
                            print(f"    Writing images directly to zarr (chunksize={chunksize})...", flush=True)
                            print(f"    [{' ' * 50}] 0%", end='', flush=True)
                        total = len(tasks)
                        completed = 0
                        write_results = []
                        for result in pool.imap_unordered(_load_and_write_to_ome_zarr_base, tasks, chunksize=chunksize):
                            write_results.append(result)
                            completed += 1
                            if verbose:
                                percent = 100 * completed // total
                                filled = int(50 * completed / total)
                                bar = '=' * filled + ' ' * (50 - filled)
                                print(f"\r    [{bar}] {percent}% ({completed}/{total})", end='', flush=True)
                        if verbose:
                            print("", flush=True)  # New line after progress
                    
                    # Check for failures
                    failures = [r for r in write_results if not r[1]]
                    if failures:
                        if verbose:
                            print(f"    WARNING: {len(failures)} images failed to write", flush=True)
                    
                    if verbose:
                        print(f"\n    ✓ Wrote {len(write_results) - len(failures)} images directly to zarr using {workers} parallel workers", flush=True)
                        try:
                            import psutil
                            current_process = psutil.Process()
                            children = current_process.children(recursive=True)
                            print(f"    DEBUG: After writing, active child processes: {len(children)}", flush=True)
                        except ImportError:
                            pass
                finally:
                    pool.close()
                    pool.join()
            else:
                # Sequential writing (small stacks or num_workers=1)
                if verbose:
                    print(f"\n    Writing images sequentially directly to zarr...", flush=True)
                    print(f"    WRITING {len(file_list)} IMAGES - PROGRESS BAR BELOW:", flush=True)
                    print("-"*70, flush=True)
                
                # Write directly to zarr sequentially
                if tqdm is not None:
                    if verbose:
                        print("", flush=True)  # Blank line before progress bar
                    for z_idx, (_, filepath) in enumerate(
                        tqdm(file_list, desc="    WRITING TO ZARR", unit="img", ncols=100, miniters=1)
                    ):
                        result = _load_and_write_to_ome_zarr_base((
                            z_idx, filepath, str(output_path), "0",
                            base_shape, dtype, has_channels, final_axis_order, C, Y, X
                        ))
                        if not result[1] and verbose:
                            print(f"    WARNING: Failed to write image {z_idx}", flush=True)
                    if verbose:
                        print("", flush=True)  # Blank line after progress bar
                else:
                    # Manual progress bar
                    total = len(file_list)
                    if verbose:
                        print(f"    [{' ' * 50}] 0%", end='', flush=True)
                    for idx, (z_idx, (_, filepath)) in enumerate(enumerate(file_list), 1):
                        result = _load_and_write_to_ome_zarr_base((
                            z_idx, filepath, str(output_path), "0",
                            base_shape, dtype, has_channels, final_axis_order, C, Y, X
                        ))
                        if verbose:
                            percent = 100 * idx // total
                            filled = int(50 * idx / total)
                            bar = '=' * filled + ' ' * (50 - filled)
                            print(f"\r    [{bar}] {percent}% ({idx}/{total})", end='', flush=True)
                    if verbose:
                        print("", flush=True)  # New line after progress
                
                if verbose:
                    print(f"    ✓ Wrote {len(file_list)} images directly to zarr", flush=True)

            # Base level is now written directly to zarr - no need to load into memory!
            # For pyramid generation, we'll read from zarr lazily using Dask
            if verbose:
                print(f"\n    Base level written directly to zarr (no memory overhead!)", flush=True)
                print(f"    Base array shape: {base_shape}", flush=True)
                print(f"    Chunk size: {base_chunks}", flush=True)
            multiscales_metadata.append(
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [1.0] * len(base_shape),  # Base level has scale 1
                        }
                    ],
                }
            )

            # Create pyramid levels using Dask for parallel processing
            if num_pyramid_levels > 1:
                try:
                    import dask
                    import dask.array as da
                except ImportError as err:
                    raise ImportError(
                        "dask is required for pyramid creation. Install with: pip install dask"
                    ) from err

                # Configure Dask for maximum parallelism
                # Use processes (not threads) to bypass GIL for CPU-bound numpy operations
                # This is critical for large arrays on multi-core machines
                # Use same worker count as image loading for consistency
                if num_workers is None:
                    num_cores = multiprocessing.cpu_count()
                elif num_workers > 1:
                    num_cores = num_workers
                else:
                    num_cores = 1
                
                if verbose:
                    print("\n" + "="*70, flush=True)
                    print(f"  [STEP 2/3] CREATING PYRAMID LEVELS - BIG LOOP STARTS NOW", flush=True)
                    print("="*70, flush=True)
                    print(f"    Creating {num_pyramid_levels - 1} pyramid level(s)...", flush=True)
                    print(f"    Dask configuration:", flush=True)
                    print(f"      Scheduler: processes (bypasses GIL)", flush=True)
                    print(f"      Workers: {num_cores} (one per core)", flush=True)
                    print(f"      Threads per worker: 1", flush=True)
                
                # Configure Dask to use all available cores with processes
                # Set chunk size based on available memory and cores
                # For 800 slices of 3k x 3k: ~800 * 3000 * 3000 * 2 bytes = ~14GB per level
                # Use larger chunks to reduce overhead, but not too large to fit in memory
                # Wrap entire pyramid creation in scheduler context
                if verbose:
                    try:
                        import psutil
                        actual_cpu_count = psutil.cpu_count(logical=False)
                        logical_cpu_count = psutil.cpu_count(logical=True)
                        print(f"    DEBUG: System has {actual_cpu_count} physical cores, {logical_cpu_count} logical cores", flush=True)
                        print(f"    DEBUG: Dask will use {num_cores} workers", flush=True)
                    except ImportError:
                        pass
                
                with dask.config.set(
                    scheduler="processes",  # Use processes, not threads (bypasses GIL)
                    num_workers=num_cores,  # Use all cores
                    threads_per_worker=1,   # One thread per worker (processes handle parallelism)
                ):
                    if verbose:
                        # Verify Dask configuration
                        print(f"    DEBUG: Dask scheduler = {dask.config.get('scheduler')}", flush=True)
                        print(f"    DEBUG: Dask num_workers = {dask.config.get('num_workers')}", flush=True)
                        print(f"    DEBUG: Dask threads_per_worker = {dask.config.get('threads_per_worker')}", flush=True)
                    # Convert base array to Dask array for efficient downsampling
                    # Use optimal chunk size: balance between parallelism and memory
                    # For 800x3000x3000: chunks of ~(50, 1000, 1000) gives ~800MB chunks, good parallelism
                    optimal_chunks = tuple(
                        min(d // 4, max(d // (num_cores * 2), 256)) if i >= len(base_shape) - 2 else d
                        for i, d in enumerate(base_shape)
                    )
                    current_dask = da.from_zarr(base_zarr_array, chunks=optimal_chunks)
                    # No need to load base_array_data into memory - read from zarr lazily!
                    current_shape = base_shape
                    # Track previous cumulative scale factors to compute incremental ones
                    prev_scale_factors = None

                    # Helper function to build zoom factors
                    def _build_zoom_factors(
                        scale_factors, has_channels, final_axis_order, current_shape
                    ):
                        """Build zoom factors for scipy.ndimage.zoom."""
                        if has_channels:
                            if final_axis_order == "ZCYX":
                                z_scale, c_scale, y_scale, x_scale = scale_factors
                                return [
                                    1.0 / z_scale,
                                    1.0,
                                    1.0 / y_scale,
                                    1.0 / x_scale,
                                ]
                            elif final_axis_order == "CZYX":
                                c_scale, z_scale, y_scale, x_scale = scale_factors
                                return [
                                    1.0,
                                    1.0 / z_scale,
                                    1.0 / y_scale,
                                    1.0 / x_scale,
                                ]
                            else:
                                z_scale, c_scale, y_scale, x_scale = scale_factors
                                return [1.0] * (len(current_shape) - 2) + [
                                    1.0 / y_scale,
                                    1.0 / x_scale,
                                ]
                        else:
                            z_scale, y_scale, x_scale = scale_factors
                            return [1.0 / z_scale, 1.0 / y_scale, 1.0 / x_scale]

                    for level_idx, cumulative_scale_factors in enumerate(
                        pyramid_scale_factors, 1
                    ):
                        if verbose:
                            level_start_time = time.time()
                            print(f"\n    Level {level_idx}/{num_pyramid_levels - 1}:", flush=True)
                            print(f"      Current shape: {current_shape}", flush=True)
                            print(f"      Cumulative scale factors: {cumulative_scale_factors}", flush=True)

                        # Compute incremental scale factors from cumulative ones
                        # Cumulative scale factors are relative to base (e.g., 2, 4, 8)
                        # But we need incremental factors relative to previous level (always 2x)
                        if prev_scale_factors is None:
                            # First level: incremental = cumulative
                            incremental_scale_factors = cumulative_scale_factors
                        else:
                            # Subsequent levels: incremental = cumulative / previous_cumulative
                            incremental_scale_factors = tuple(
                                curr / prev if prev > 0 else curr
                                for curr, prev in zip(
                                    cumulative_scale_factors, prev_scale_factors
                                )
                            )
                        
                        if verbose:
                            print(f"      Incremental scale factors: {incremental_scale_factors}", flush=True)

                        # Determine which dimensions to downsample based on incremental scale factors
                        # Build coarsen dictionary: {axis_index: scale_factor}
                        coarsen_dict = {}

                        if has_channels:
                            if final_axis_order == "ZCYX":
                                # Shape: (Z, C, Y, X), incremental_scale_factors: (z_scale, c_scale, y_scale, x_scale)
                                (
                                    z_scale,
                                    c_scale,
                                    y_scale,
                                    x_scale,
                                ) = incremental_scale_factors
                                if z_scale > 1:
                                    coarsen_dict[0] = int(z_scale)  # Z axis
                                # Skip C axis (axis 1) - never downsample channels
                                if y_scale > 1:
                                    coarsen_dict[2] = int(y_scale)  # Y axis
                                if x_scale > 1:
                                    coarsen_dict[3] = int(x_scale)  # X axis
                            elif final_axis_order == "CZYX":
                                # Shape: (C, Z, Y, X), incremental_scale_factors: (c_scale, z_scale, y_scale, x_scale)
                                (
                                    c_scale,
                                    z_scale,
                                    y_scale,
                                    x_scale,
                                ) = incremental_scale_factors
                                # Skip C axis (axis 0) - never downsample channels
                                if z_scale > 1:
                                    coarsen_dict[1] = int(z_scale)  # Z axis
                                if y_scale > 1:
                                    coarsen_dict[2] = int(y_scale)  # Y axis
                                if x_scale > 1:
                                    coarsen_dict[3] = int(x_scale)  # X axis
                            else:
                                # Generic: assume standard order and downsample based on scale factors
                                # This is a fallback - ideally users should specify correct axis order
                                for dim_idx, scale in enumerate(incremental_scale_factors):
                                    if (
                                        scale > 1 and dim_idx != 1
                                    ):  # Don't downsample channels
                                        coarsen_dict[dim_idx] = int(scale)
                        else:
                            # Single channel: (Z, Y, X), incremental_scale_factors: (z_scale, y_scale, x_scale)
                            z_scale, y_scale, x_scale = incremental_scale_factors
                            if z_scale > 1:
                                coarsen_dict[0] = int(z_scale)  # Z axis
                            if y_scale > 1:
                                coarsen_dict[1] = int(y_scale)  # Y axis
                            if x_scale > 1:
                                coarsen_dict[2] = int(x_scale)  # X axis

                        # Downsample using Dask coarsen (block averaging)
                        if downsample_method == "dask_coarsen":
                            if coarsen_dict:
                                if verbose:
                                    print(f"      Downsampling axes: {list(coarsen_dict.keys())} with factors {list(coarsen_dict.values())}", flush=True)
                                # Check if dimensions are divisible by scale factors
                                # Dask coarsen requires exact divisibility
                                # If not divisible, pad with zeros to make them divisible
                                # Padding is done at the end (right/bottom) of each dimension
                                needs_padding = False
                                padded_shape = list(current_shape)
                                pad_widths = [(0, 0)] * len(current_shape)

                                for axis, scale in coarsen_dict.items():
                                    if current_shape[axis] % scale != 0:
                                        needs_padding = True
                                        # Calculate padding needed to make divisible
                                        remainder = current_shape[axis] % scale
                                        padding_needed = scale - remainder
                                        padded_shape[axis] = (
                                            current_shape[axis] + padding_needed
                                        )
                                        # Pad at the end (right/bottom) - zeros are added to edges
                                        pad_widths[axis] = (0, padding_needed)

                                if needs_padding:
                                    if verbose:
                                        print(f"      Padding required: {padded_shape} (from {current_shape})", flush=True)
                                    # Pad the Dask array with zeros (lazy operation)
                                    # Convert pad_widths to Dask format: list of (before, after) tuples
                                    current_dask_padded = da.pad(
                                        current_dask,
                                        pad_widths,
                                        mode="constant",
                                        constant_values=0,
                                    )
                                else:
                                    current_dask_padded = current_dask

                                # Use dask coarsen with mean reduction for block averaging
                                if verbose:
                                    print(f"      Computing downsampled array using Dask (processes scheduler)...", flush=True)
                                    print(f"      This may take a while for large arrays...", flush=True)
                                    # Show task graph info
                                    downsampled_dask = da.coarsen(
                                        np.mean, current_dask_padded, coarsen_dict
                                    )
                                    num_tasks = len(downsampled_dask.dask)
                                    print(f"      Dask task graph: {num_tasks} tasks", flush=True)
                                    print(f"      Array chunks: {downsampled_dask.numblocks}", flush=True)
                                    print(f"      Starting parallel computation with {num_cores} workers...", flush=True)
                                else:
                                    downsampled_dask = da.coarsen(
                                        np.mean, current_dask_padded, coarsen_dict
                                    )
                                
                                # Compute the result using process scheduler (configured above)
                                # This will use all available cores efficiently
                                # Add progress callback if verbose
                                if verbose:
                                    print(f"\n      COMPUTING DOWNSAMPLED ARRAY - PROGRESS BAR BELOW:", flush=True)
                                    print("-"*70, flush=True)
                                    # Monitor CPU usage before computation
                                    try:
                                        import psutil
                                        cpu_before = psutil.cpu_percent(interval=0.1, percpu=True)
                                        active_cores_before = sum(1 for c in cpu_before if c > 5)
                                        print(f"      DEBUG: Active CPU cores before compute: {active_cores_before}/{len(cpu_before)}", flush=True)
                                    except ImportError:
                                        pass
                                    
                                    try:
                                        from dask.diagnostics import ProgressBar
                                        print("", flush=True)  # Blank line before progress bar
                                        with ProgressBar():
                                            downsampled = downsampled_dask.compute().astype(dtype)
                                        print("", flush=True)  # Blank line after progress bar
                                    except ImportError:
                                        # Fallback: manual progress updates
                                        print(f"      Processing {num_tasks} tasks...", flush=True)
                                        downsampled = downsampled_dask.compute().astype(dtype)
                                    
                                    # Monitor CPU usage after computation
                                    try:
                                        import psutil
                                        cpu_after = psutil.cpu_percent(interval=0.1, percpu=True)
                                        active_cores_after = sum(1 for c in cpu_after if c > 5)
                                        print(f"      DEBUG: Active CPU cores after compute: {active_cores_after}/{len(cpu_after)}", flush=True)
                                        # Check active processes
                                        current_process = psutil.Process()
                                        children = current_process.children(recursive=True)
                                        print(f"      DEBUG: Active child processes during compute: {len(children)}", flush=True)
                                    except ImportError:
                                        pass
                                else:
                                    downsampled = downsampled_dask.compute().astype(dtype)
                                
                                if verbose:
                                    print(f"      ✓ Downsampled shape: {downsampled.shape}", flush=True)
                            else:
                                # No downsampling needed for this level (shouldn't happen, but handle gracefully)
                                # Read from current_dask
                                downsampled = current_dask.compute().astype(dtype)

                        elif downsample_method == "scipy_zoom":
                            # Use scipy zoom (interpolation-based downsampling)
                            try:
                                from scipy.ndimage import zoom
                            except ImportError as err:
                                raise ImportError(
                                    "scipy is required for scipy_zoom method. Install with: pip install scipy"
                                ) from err

                            zoom_factors = _build_zoom_factors(
                                incremental_scale_factors,
                                has_channels,
                                final_axis_order,
                                current_shape,
                            )
                            # Read from current_dask for scipy zoom
                            current_data = current_dask.compute()
                            downsampled = zoom(
                                current_data, zoom_factors, order=1, prefilter=False
                            ).astype(dtype)

                        else:
                            raise ValueError(
                                f"Unknown downsample_method: {downsample_method}. "
                                "Supported methods: 'dask_coarsen', 'scipy_zoom'"
                            )

                        # Update shape for next iteration
                        current_shape = downsampled.shape

                        # Create array for this pyramid level (stored at root)
                        if verbose:
                            print(f"      Writing level {level_idx} to zarr...", flush=True)
                        level_zarr_array = _create_zarr_array(
                            root,
                            str(level_idx),
                            data=downsampled,
                            chunks=tuple(min(d, 256) for d in current_shape),
                        )
                        if verbose:
                            level_time = time.time() - level_start_time
                            print(f"      ✓ Level {level_idx} complete ({level_time:.1f}s)", flush=True)
                        multiscales_metadata.append(
                            {
                                "path": str(level_idx),
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": list(cumulative_scale_factors),
                                    }
                                ],
                            }
                        )

                        # Update previous scale factors for next iteration
                        prev_scale_factors = cumulative_scale_factors

                        # Convert to Dask for next iteration (if there are more levels)
                        if level_idx < len(pyramid_scale_factors):
                            # Use optimal chunks for next level too
                            next_shape = current_shape
                            next_optimal_chunks = tuple(
                                min(d // 4, max(d // (num_cores * 2), 256)) if i >= len(next_shape) - 2 else d
                                for i, d in enumerate(next_shape)
                            )
                            current_dask = da.from_zarr(level_zarr_array, chunks=next_optimal_chunks)

            # Create OME metadata
            if verbose:
                print(f"\n  [Step 3/3] Writing OME metadata...", flush=True)
            # Determine axis names based on shape
            if has_channels:
                if final_axis_order == "ZCYX":
                    axes = ["z", "c", "y", "x"]
                elif final_axis_order == "CZYX":
                    axes = ["c", "z", "y", "x"]
                else:
                    axes = ["z", "c", "y", "x"]  # Default
            else:
                axes = ["z", "y", "x"]

            ome_metadata = {
                "multiscales": [
                    {
                        "version": "0.4",
                        "axes": [
                            {
                                "name": ax,
                                "type": "space" if ax in ["x", "y", "z"] else "channel",
                            }
                            for ax in axes
                        ],
                        "datasets": multiscales_metadata,
                    }
                ]
            }

            # Add OME metadata to root
            root.attrs["multiscales"] = ome_metadata["multiscales"]
            root.attrs["omero"] = {
                "id": 1,
                "name": basename,
                "version": "0.4",
            }

            # Store additional metadata
            root.attrs["basename"] = basename
            root.attrs["file_count"] = len(file_list)
            root.attrs["counter_range"] = [counter_min, counter_max]
            root.attrs["axis_order"] = final_axis_order
            root.attrs["files"] = [str(f) for _, f in file_list]
            root.attrs["pattern"] = (
                pattern.pattern if isinstance(pattern, re.Pattern) else pattern
            )
            root.attrs["extension"] = extension

            if verbose:
                print(f"\n  ✓ Completed OME-Zarr: {basename}", flush=True)
                print(f"  Output: {output_path}", flush=True)
                print(f"  Total pyramid levels: {num_pyramid_levels}", flush=True)
                print(f"{'='*70}", flush=True)
        else:
            print(f"  Dry run: Would create OME-Zarr at {output_path}")
            print(f"  Base shape: {base_shape}, dtype: {dtype}")
            print(f"  Pyramid levels: {num_pyramid_levels}")

        # Store results
        results[basename] = {
            "zarr_path": str(output_path),
            "shape": base_shape,
            "dtype": dtype,
            "file_count": len(file_list),
            "files": [str(f) for _, f in file_list],
            "counter_range": (counter_min, counter_max),
            "axis_order": final_axis_order,
            "pyramid_levels": num_pyramid_levels,
        }

    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ Successfully processed {len(results)} stack(s) as OME-Zarr")
        for stack_name, metadata in results.items():
            print(f"  - {stack_name}: {metadata['zarr_path']}")
            print(f"    Shape: {metadata['shape']}, Levels: {metadata['pyramid_levels']}")
    return results
