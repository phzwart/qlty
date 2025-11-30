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


def _load_and_write_to_all_pyramid_levels(
    args: tuple,
) -> tuple[int, bool]:
    """
    Load an image, downsample it progressively, and write to ALL pyramid levels in one pass.

    This is MUCH more efficient than writing base level then reading back for downsampling.
    For 2D downsampling mode, we downsample each slice independently (Y, X only).

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - z_idx: int - Z-index in the zarr array
        - filepath: Path - Path to image file
        - zarr_group_path: str - Path to zarr group (OME-Zarr root)
        - pyramid_level_shapes: list[tuple] - Shapes for each pyramid level
        - pyramid_scale_factors: list[tuple] - Cumulative scale factors for each level
        - dtype: np.dtype - Target dtype
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
        pyramid_level_shapes,
        pyramid_scale_factors,
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

        # Open zarr group (read-write mode supports concurrent writes)
        zarr_group = zarr.open_group(zarr_group_path, mode="r+")

        # Apply axis order transformation if needed
        if has_channels:
            # We have img as (C, Y, X), need to write at z_idx
            # Create a (1, C, Y, X) array, apply transformation
            slice_data = img[np.newaxis, ...]  # (1, C, Y, X)
            slice_reordered, _ = _apply_axis_order(
                slice_data,
                (1, C, Y, X),
                axis_order,
            )
            img_reordered = slice_reordered[0]  # Remove Z dimension, now (C, Y, X) or reordered
        else:
            img_reordered = img  # (Y, X)

        # Write to base level (level 0)
        base_array = zarr_group["0"]
        if has_channels:
            if axis_order == "CZYX":
                base_array[:, z_idx, :, :] = img_reordered
            elif axis_order == "ZCYX":
                base_array[z_idx, :, :, :] = img_reordered
            else:
                # Generic: assume Z is first dimension
                base_array[z_idx, ...] = img_reordered
        else:
            base_array[z_idx, :, :] = img_reordered

        # Now downsample progressively and write to each pyramid level
        # For 2D mode, we downsample Y and X dimensions only
        current_img = img_reordered.copy()
        prev_scale_factors = None

        for level_idx, (expected_level_shape, cumulative_scale_factors) in enumerate(
            zip(pyramid_level_shapes[1:], pyramid_scale_factors), start=1
        ):
            # Calculate incremental scale factors
            if prev_scale_factors is None:
                incremental_scale_factors = cumulative_scale_factors
            else:
                incremental_scale_factors = tuple(
                    curr / prev if prev > 0 else curr
                    for curr, prev in zip(cumulative_scale_factors, prev_scale_factors)
                )

            # Extract Y, X scale factors (for 2D downsampling)
            # For 2D mode, we only downsample spatial dimensions (Y, X)
            if has_channels:
                # Extract Y, X from scale factors (last two dimensions)
                if len(incremental_scale_factors) == 4:
                    # (Z, C, Y, X) or (C, Z, Y, X) - take last two
                    y_scale, x_scale = incremental_scale_factors[-2:]
                else:
                    y_scale, x_scale = incremental_scale_factors[-2:]
            else:
                # Single channel: (Z, Y, X) - take last two
                y_scale, x_scale = incremental_scale_factors[-2:]

            # Downsample using block averaging with padding if needed
            y_scale_int = int(y_scale)
            x_scale_int = int(x_scale)

            if has_channels:
                # Image is (C, Y, X)
                C_dim, Y_dim, X_dim = current_img.shape

                # Pad if needed to make divisible
                pad_Y = (y_scale_int - (Y_dim % y_scale_int)) % y_scale_int
                pad_X = (x_scale_int - (X_dim % x_scale_int)) % x_scale_int

                if pad_Y > 0 or pad_X > 0:
                    padded = np.pad(
                        current_img,
                        ((0, 0), (0, pad_Y), (0, pad_X)),
                        mode="constant",
                        constant_values=0,
                    )
                    Y_padded = Y_dim + pad_Y
                    X_padded = X_dim + pad_X
                else:
                    padded = current_img
                    Y_padded = Y_dim
                    X_padded = X_dim

                # Block average downsampling
                downsampled = (
                    padded.reshape(C_dim, Y_padded // y_scale_int, y_scale_int, X_padded // x_scale_int, x_scale_int)
                    .mean(axis=(2, 4))
                    .astype(dtype)
                )
            else:
                # Single channel: (Y, X)
                Y_dim, X_dim = current_img.shape

                # Pad if needed
                pad_Y = (y_scale_int - (Y_dim % y_scale_int)) % y_scale_int
                pad_X = (x_scale_int - (X_dim % x_scale_int)) % x_scale_int

                if pad_Y > 0 or pad_X > 0:
                    padded = np.pad(
                        current_img,
                        ((0, pad_Y), (0, pad_X)),
                        mode="constant",
                        constant_values=0,
                    )
                    Y_padded = Y_dim + pad_Y
                    X_padded = X_dim + pad_X
                else:
                    padded = current_img
                    Y_padded = Y_dim
                    X_padded = X_dim

                # Block average downsampling
                downsampled = (
                    padded.reshape(Y_padded // y_scale_int, y_scale_int, X_padded // x_scale_int, x_scale_int)
                    .mean(axis=(1, 3))
                    .astype(dtype)
                )

            # Write downsampled image to this pyramid level
            level_array = zarr_group[str(level_idx)]
            
            # Validate and write - expected_level_shape is from the loop iteration
            if has_channels:
                # downsampled shape is (C, Y, X)
                # We need to match the spatial dimensions (Y, X) from expected_level_shape
                if axis_order == "CZYX":
                    # Array shape: (C, Z, Y, X)
                    # Expected: (C, Z, Y, X) -> slice at z_idx should be (C, Y, X)
                    expected_C, expected_Z, expected_Y, expected_X = expected_level_shape
                    C_actual, Y_actual, X_actual = downsampled.shape
                    
                    # Fix shape if needed
                    if C_actual != expected_C or Y_actual != expected_Y or X_actual != expected_X:
                        if C_actual > expected_C or Y_actual > expected_Y or X_actual > expected_X:
                            downsampled = downsampled[:expected_C, :expected_Y, :expected_X]
                        elif C_actual < expected_C or Y_actual < expected_Y or X_actual < expected_X:
                            padded = np.zeros((expected_C, expected_Y, expected_X), dtype=downsampled.dtype)
                            padded[:C_actual, :Y_actual, :X_actual] = downsampled
                            downsampled = padded
                    
                    level_array[:, z_idx, :, :] = downsampled
                elif axis_order == "ZCYX":
                    # Array shape: (Z, C, Y, X)
                    # Expected: (Z, C, Y, X) -> slice at z_idx should be (C, Y, X)
                    expected_Z, expected_C, expected_Y, expected_X = expected_level_shape
                    C_actual, Y_actual, X_actual = downsampled.shape
                    
                    # Fix shape if needed
                    if C_actual != expected_C or Y_actual != expected_Y or X_actual != expected_X:
                        if C_actual > expected_C or Y_actual > expected_Y or X_actual > expected_X:
                            downsampled = downsampled[:expected_C, :expected_Y, :expected_X]
                        elif C_actual < expected_C or Y_actual < expected_Y or X_actual < expected_X:
                            padded = np.zeros((expected_C, expected_Y, expected_X), dtype=downsampled.dtype)
                            padded[:C_actual, :Y_actual, :X_actual] = downsampled
                            downsampled = padded
                    
                    level_array[z_idx, :, :, :] = downsampled
                else:
                    # Generic: assume Z is first dimension
                    # Expected level shape should match array shape
                    expected_slice_shape = expected_level_shape[1:]  # Skip Z dimension
                    if downsampled.shape != expected_slice_shape:
                        # Try to fix shape
                        if len(downsampled.shape) == len(expected_slice_shape):
                            # Same dimensionality, try crop/pad
                            fixed = np.zeros(expected_slice_shape, dtype=downsampled.dtype)
                            slices = tuple(slice(0, min(d1, d2)) for d1, d2 in zip(downsampled.shape, expected_slice_shape))
                            fixed[slices] = downsampled[slices]
                            downsampled = fixed
                        else:
                            raise ValueError(
                                f"Shape mismatch at level {level_idx}, z_idx {z_idx}: "
                                f"downsampled shape {downsampled.shape} != expected {expected_slice_shape}. "
                                f"Level array shape: {level_array.shape}, expected level shape: {expected_level_shape}"
                            )
                    level_array[z_idx, ...] = downsampled
            else:
                # Single channel: downsampled is (Y, X)
                # Expected level shape: (Z, Y, X) -> slice should be (Y, X)
                expected_Z, expected_Y, expected_X = expected_level_shape
                
                # Verify array shape matches expected
                if level_array.shape != expected_level_shape:
                    raise ValueError(
                        f"Array shape mismatch at level {level_idx}: "
                        f"array shape {level_array.shape} != expected {expected_level_shape}"
                    )
                
                # Verify downsampled shape matches expected slice
                if downsampled.shape != (expected_Y, expected_X):
                    raise ValueError(
                        f"Shape mismatch at level {level_idx}, z_idx {z_idx}: "
                        f"downsampled shape {downsampled.shape} != expected (Y={expected_Y}, X={expected_X}). "
                        f"Level array shape: {level_array.shape}, expected level shape: {expected_level_shape}. "
                        f"Current image shape before downsampling: {current_img.shape}"
                    )
                
                # Ensure we're writing the exact shape expected
                # If shapes don't match, crop or pad to match exactly
                Y_actual, X_actual = downsampled.shape
                if Y_actual != expected_Y or X_actual != expected_X:
                    if Y_actual > expected_Y or X_actual > expected_X:
                        # Crop to expected size
                        downsampled = downsampled[:expected_Y, :expected_X]
                    elif Y_actual < expected_Y or X_actual < expected_X:
                        # Pad to expected size
                        padded = np.zeros((expected_Y, expected_X), dtype=downsampled.dtype)
                        padded[:Y_actual, :X_actual] = downsampled
                        downsampled = padded
                
                # Final verification
                assert downsampled.shape == (expected_Y, expected_X), \
                    f"Shape fix failed: {downsampled.shape} != ({expected_Y}, {expected_X})"
                
                # Write with exact shape match
                level_array[z_idx, :, :] = downsampled

            # Update for next level
            current_img = downsampled
            prev_scale_factors = cumulative_scale_factors

        return (z_idx, True)
    except Exception:
        import traceback
        traceback.print_exc()
        return (z_idx, False)


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
            # Validate downsample_method
            # Note: Currently only immediate downsampling (block averaging) is supported
            # The downsample_method parameter is kept for API compatibility but not used
            if downsample_method not in ("dask_coarsen", "scipy_zoom"):
                raise ValueError(
                    f"Unknown downsample_method: {downsample_method}. "
                    "Supported methods: 'dask_coarsen', 'scipy_zoom'. "
                    "Note: Currently all methods use immediate block-averaging downsampling."
                )

            # Validate downsample_mode - 3D mode not yet implemented
            if downsample_mode == "3d":
                raise NotImplementedError(
                    "3D downsampling mode is not yet implemented. "
                    "Currently only 2D downsampling (Y, X axes) is supported."
                )

            if verbose:
                print(f"  Creating OME-Zarr: {output_path}", flush=True)
                print(f"  Base shape: {base_shape}, dtype: {dtype}", flush=True)
                print(f"  Pyramid levels: {num_pyramid_levels}", flush=True)
                print(f"  Downsample method: {downsample_method}", flush=True)
                print(f"  Downsample mode: {downsample_mode}", flush=True)
                print("\n  *** STARTING PROCESSING - THIS MAY TAKE A WHILE ***", flush=True)
                print("  *** WATCH FOR PROGRESS BARS BELOW ***\n", flush=True)

            # Create OME-Zarr root group
            if verbose:
                print("  Creating zarr root group...", flush=True)
            root = zarr.open_group(str(output_path), mode="w")
            multiscales_metadata = []
            if verbose:
                print("  ✓ Zarr root group created", flush=True)
                print("  Calculating pyramid level shapes...", flush=True)

            # Calculate pyramid level shapes progressively, accounting for padding
            # This MUST match the exact downsampling process: pad -> downsample
            # We simulate the progressive downsampling to get exact shapes
            pyramid_level_shapes = [base_shape]
            if num_pyramid_levels > 1:
                # Track current shape as we progressively downsample (simulating the process)
                current_simulated_shape = list(base_shape)
                prev_cumulative_scale_factors = None
                
                for cumulative_scale_factors in pyramid_scale_factors:
                    # Calculate incremental scale factors (same as in downsampling)
                    if prev_cumulative_scale_factors is None:
                        incremental_scale_factors = cumulative_scale_factors
                    else:
                        incremental_scale_factors = tuple(
                            curr / prev if prev > 0 else curr
                            for curr, prev in zip(cumulative_scale_factors, prev_cumulative_scale_factors)
                        )
                    
                    # Extract Y, X scale factors for 2D downsampling
                    if has_channels:
                        # Extract Y, X from scale factors (last two dimensions)
                        y_scale = incremental_scale_factors[-2]
                        x_scale = incremental_scale_factors[-1]
                        # Get current Y, X dimensions (last two)
                        if final_axis_order == "ZCYX":
                            # Shape: (Z, C, Y, X)
                            Y_dim = current_simulated_shape[2]
                            X_dim = current_simulated_shape[3]
                        elif final_axis_order == "CZYX":
                            # Shape: (C, Z, Y, X)
                            Y_dim = current_simulated_shape[2]
                            X_dim = current_simulated_shape[3]
                        else:
                            # Generic: assume Y, X are last two
                            Y_dim = current_simulated_shape[-2]
                            X_dim = current_simulated_shape[-1]
                    else:
                        # Single channel: (Z, Y, X)
                        y_scale, x_scale = incremental_scale_factors[-2:]
                        Y_dim = current_simulated_shape[1]
                        X_dim = current_simulated_shape[2]
                    
                    # Calculate padding (same logic as actual downsampling)
                    y_scale_int = int(y_scale)
                    x_scale_int = int(x_scale)
                    pad_Y = (y_scale_int - (Y_dim % y_scale_int)) % y_scale_int
                    pad_X = (x_scale_int - (X_dim % x_scale_int)) % x_scale_int
                    
                    # Calculate new dimensions after padding and downsampling
                    Y_padded = Y_dim + pad_Y
                    X_padded = X_dim + pad_X
                    Y_new = Y_padded // y_scale_int
                    X_new = X_padded // x_scale_int
                    
                    # Build new level shape
                    if has_channels:
                        if final_axis_order == "ZCYX":
                            level_shape = (
                                current_simulated_shape[0],  # Z unchanged
                                current_simulated_shape[1],  # C unchanged
                                Y_new,
                                X_new,
                            )
                        elif final_axis_order == "CZYX":
                            level_shape = (
                                current_simulated_shape[0],  # C unchanged
                                current_simulated_shape[1],  # Z unchanged
                                Y_new,
                                X_new,
                            )
                        else:
                            # Generic: keep all dims except last two
                            level_shape = tuple(current_simulated_shape[:-2]) + (Y_new, X_new)
                    else:
                        # Single channel: (Z, Y, X)
                        level_shape = (
                            current_simulated_shape[0],  # Z unchanged for 2D mode
                            Y_new,
                            X_new,
                        )
                    
                    pyramid_level_shapes.append(level_shape)
                    
                    # Update simulated shape for next iteration
                    current_simulated_shape = list(level_shape)
                    prev_cumulative_scale_factors = cumulative_scale_factors
            
            if verbose:
                print(f"  ✓ Calculated {len(pyramid_level_shapes)} pyramid level shapes", flush=True)
                for idx, shape in enumerate(pyramid_level_shapes):
                    print(f"    Level {idx}: {shape}", flush=True)

            # Determine chunk size for all levels
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

            # Create ALL pyramid level arrays upfront (empty, we'll write to them in parallel)
            # Use zarr 3.0+ API: shape must be a keyword argument
            if verbose:
                print("\n    Creating all pyramid level zarr arrays (empty, will write in parallel)...", flush=True)
                print(f"    Creating base level (0) with shape {base_shape}...", flush=True)
            base_zarr_array = root.create(
                "0",
                shape=base_shape,
                chunks=base_chunks,
                dtype=dtype,
            )
            pyramid_zarr_arrays = [base_zarr_array]
            if verbose:
                print(f"    ✓ Created base level (0)", flush=True)

            # Create pyramid level arrays
            for level_idx, level_shape in enumerate(pyramid_level_shapes[1:], start=1):
                if verbose:
                    print(f"    Creating pyramid level {level_idx} with shape {level_shape}...", flush=True)
                level_chunks = tuple(min(d, 256) for d in level_shape)
                # Zarr 3.0+ API: shape must be a keyword argument
                level_array = root.create(
                    str(level_idx),
                    shape=level_shape,
                    chunks=level_chunks,
                    dtype=dtype,
                )
                pyramid_zarr_arrays.append(level_array)
                if verbose:
                    print(f"    ✓ Created pyramid level {level_idx}", flush=True)

            if verbose:
                print(f"    ✓ Created {len(pyramid_level_shapes)} pyramid level arrays", flush=True)
                print("\n" + "="*70, flush=True)
                print("  [STEP 1/1] LOADING + DOWNSAMPLING + WRITING TO ALL PYRAMID LEVELS - ULTRA FAST MODE", flush=True)
                print("="*70 + "\n", flush=True)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()

            # Setup multiprocessing
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
                    print(f"    Processing {len(file_list)} images: load → downsample → write to all {num_pyramid_levels} levels", flush=True)
                else:
                    print("    Using sequential processing (1 worker)", flush=True)
                    print(f"    Processing {len(file_list)} images...", flush=True)

            # Use immediate downsampling: load image, downsample progressively, write to all levels
            if use_multiprocessing and len(file_list) > 10:
                # Prepare tasks for parallel load-and-write with immediate downsampling
                tasks = []
                for z_idx, (_, filepath) in enumerate(file_list):
                    tasks.append(
                        (
                            z_idx,
                            filepath,
                            str(output_path),  # zarr group path
                            pyramid_level_shapes,  # All pyramid level shapes
                            pyramid_scale_factors,  # Cumulative scale factors
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
                                pool.imap_unordered(_load_and_write_to_all_pyramid_levels, tasks, chunksize=chunksize),
                                total=len(tasks),
                                desc="    LOAD+DOWNSAMPLE+WRITE",
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
                            print(f"    Processing images with immediate downsampling (chunksize={chunksize})...", flush=True)
                            print(f"    [{' ' * 50}] 0%", end='', flush=True)
                        total = len(tasks)
                        completed = 0
                        write_results = []
                        for result in pool.imap_unordered(_load_and_write_to_all_pyramid_levels, tasks, chunksize=chunksize):
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
                    print("\n    Processing images sequentially with immediate downsampling...", flush=True)
                    print(f"    PROCESSING {len(file_list)} IMAGES - PROGRESS BAR BELOW:", flush=True)
                    print("-"*70, flush=True)

                # Write directly to zarr sequentially with immediate downsampling
                if tqdm is not None:
                    if verbose:
                        print("", flush=True)  # Blank line before progress bar
                    for z_idx, (_, filepath) in enumerate(
                        tqdm(file_list, desc="    LOAD+DOWNSAMPLE+WRITE", unit="img", ncols=100, miniters=1)
                    ):
                        result = _load_and_write_to_all_pyramid_levels((
                            z_idx, filepath, str(output_path),
                            pyramid_level_shapes, pyramid_scale_factors,
                            dtype, has_channels, final_axis_order, C, Y, X
                        ))
                        if not result[1] and verbose:
                            print(f"    WARNING: Failed to write image {z_idx}", flush=True)
                    if verbose:
                        print("", flush=True)  # Blank line after progress bar
                else:
                    # Manual progress bar when tqdm not available
                    if verbose:
                        print("    Processing images with immediate downsampling...", flush=True)
                        print(f"    [{' ' * 50}] 0%", end='', flush=True)
                    total = len(file_list)
                    completed = 0
                    for _idx, (z_idx, (_, filepath)) in enumerate(enumerate(file_list), 1):
                        result = _load_and_write_to_all_pyramid_levels((
                            z_idx, filepath, str(output_path),
                            pyramid_level_shapes, pyramid_scale_factors,
                            dtype, has_channels, final_axis_order, C, Y, X
                        ))
                        completed += 1
                        if verbose:
                            percent = 100 * completed // total
                            filled = int(50 * completed / total)
                            bar = '=' * filled + ' ' * (50 - filled)
                            print(f"\r    [{bar}] {percent}% ({completed}/{total})", end='', flush=True)
                    if verbose:
                        print("", flush=True)  # New line after progress

                if verbose:
                    print(f"    ✓ Processed {len(file_list)} images with immediate downsampling", flush=True)

            # All pyramid levels are now written with immediate downsampling!
            # Build metadata for all levels
            if verbose:
                print(f"\n    ✓ All {num_pyramid_levels} pyramid levels written with immediate downsampling!", flush=True)
                print(f"    Base array shape: {base_shape}", flush=True)
                print(f"    Chunk size: {base_chunks}", flush=True)

            # Add metadata for base level
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

            # Add metadata for all pyramid levels
            for level_idx, cumulative_scale_factors in enumerate(pyramid_scale_factors, start=1):
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

            # Create OME metadata
            if verbose:
                print("\n  [Step 3/3] Writing OME metadata...", flush=True)
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
