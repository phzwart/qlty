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
import sys
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

    # Try to import ProcessSynchronizer - it may be in different locations depending on zarr version
    try:
        from zarr.sync import ProcessSynchronizer
    except ImportError:
        try:
            from zarr import ProcessSynchronizer
        except ImportError:
            # ProcessSynchronizer not available - we'll handle this gracefully
            ProcessSynchronizer = None
except ImportError as err:
    msg = "zarr is required. Install with: pip install zarr"
    raise ImportError(msg) from err

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    F = None


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
        If 'compressor' is not provided, defaults to Zstd(level=1) for float32,
        or Blosc(clevel=5, codec='lz4') as fallback

    Returns
    -------
    zarr.Array
        The created zarr array
    """
    # Extract data if provided
    data = kwargs.pop("data", None)

    # Note: Compression configuration removed for Zarr 3.x compatibility
    # Zarr 3.x requires proper codec configuration with both ArrayBytesCodec and BytesBytesCodec
    # For now, we use default compression. Compression can be added via kwargs if needed.
    # The default zarr compression should be sufficient for most use cases.

    # If data is provided, extract shape and dtype for zarr 3.0.0a5 compatibility
    # zarr 3.0.0a5's create() calls create_array() which requires shape as keyword-only arg
    if data is not None:
        kwargs["shape"] = data.shape
        kwargs["dtype"] = data.dtype
        arr = group.create(name, **kwargs)
        arr[:] = data
        return arr

    return group.create(name, **kwargs)


def _get_zarr_group_keys(group):
    """
    Get keys from a zarr group in a version-compatible way.

    Works with both zarr 3.0.0a5 (which may not have keys() method or __iter__)
    and zarr 3.1.5+ (which has keys() method).

    Parameters
    ----------
    group : zarr.Group
        The zarr group

    Returns
    -------
    list
        List of keys in the group
    """
    # In zarr 3.0.0a5, groups might not have keys() method and __iter__ raises NotImplementedError
    # Try keys() first (for zarr 3.1.5+)
    if hasattr(group, "keys"):
        try:
            return list(group.keys())
        except (AttributeError, TypeError):
            pass

    # Try items() method if available
    if hasattr(group, "items"):
        try:
            return [k for k, v in group.items()]
        except (AttributeError, TypeError):
            pass

    # Try accessing internal _keys attribute (some zarr versions)
    if hasattr(group, "_keys"):
        try:
            return list(group._keys)
        except (AttributeError, TypeError):
            pass

    # Try accessing via store (zarr groups have a store attribute)
    # This works for zarr 3.0.0a5 where keys() and __iter__ don't work
    if hasattr(group, "store"):
        try:
            store = group.store
            # Get the group path
            group_path = getattr(group, "path", "") or ""
            if group_path and not group_path.endswith("/"):
                group_path += "/"

            # Try to list keys from store
            if hasattr(store, "keys"):
                all_store_keys = list(store.keys())
                # Filter keys that belong to this group
                group_keys = set()
                for store_key in all_store_keys:
                    # Check if this key belongs to our group
                    if store_key.startswith(group_path):
                        # Remove group path prefix
                        relative_key = store_key[len(group_path) :]
                        if relative_key:
                            # Split by / to get immediate children
                            parts = relative_key.split("/")
                            if len(parts) > 0:
                                # First part is the immediate child name
                                child_name = parts[0]
                                # Remove .zarray/.zgroup suffix if present
                                if child_name.endswith(
                                    ".zarray"
                                ) or child_name.endswith(".zgroup"):
                                    child_name = child_name[
                                        :-7
                                    ]  # Remove .zarray/.zgroup
                                if child_name:
                                    group_keys.add(child_name)

                if group_keys:
                    return sorted(group_keys)
        except (AttributeError, TypeError, KeyError, ValueError):
            pass

    # Last resort: try iterating (will fail in zarr 3.0.0a5 with NotImplementedError)
    try:
        return list(group)
    except (TypeError, NotImplementedError):
        # For zarr 3.0.0a5, we need to use a different approach
        # Try to access keys by checking what's accessible via __getitem__
        # This is a fallback that tries common key patterns
        keys = []
        # Try numeric keys (for pyramid levels)
        for i in range(10):  # Check levels 0-9
            try:
                _ = group[str(i)]
                keys.append(str(i))
            except (KeyError, TypeError):
                pass
        # Try diff_ keys
        for i in range(10):  # Check diff_0 to diff_9
            try:
                _ = group[f"diff_{i}"]
                keys.append(f"diff_{i}")
            except (KeyError, TypeError):
                pass
        if keys:
            return keys

        # If all else fails, raise an error
        msg = (
            "Unable to get keys from zarr group. "
            "Zarr version may be incompatible. "
            "Please upgrade to zarr >= 3.1.0 or use a different zarr version."
        )
        raise RuntimeError(msg) from None


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


def _normalize_image(
    img: np.ndarray,
    normalize: bool = False,
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """
    Normalize an image using mean subtraction and division by standard deviation.

    Parameters
    ----------
    img : np.ndarray
        Image array to normalize
    normalize : bool
        Whether to normalize the image
    mean : float | None
        Mean value for normalization. If None and normalize=True, uses image mean.
    std : float | None
        Standard deviation for normalization. If None and normalize=True, uses image std.

    Returns
    -------
    np.ndarray
        Normalized image array (same dtype as input)
    """
    if not normalize:
        return img

    # Compute mean and std if not provided
    if mean is None:
        mean = float(np.mean(img))
    if std is None:
        std = float(np.std(img))

    # Avoid division by zero
    if std == 0:
        std = 1.0

    # Normalize: (img - mean) / std
    normalized = (img.astype(np.float32) - mean) / std

    # Preserve original dtype if possible
    return normalized.astype(img.dtype)


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


def _downsample_with_torch(
    img: np.ndarray,
    y_scale: int,
    x_scale: int,
) -> np.ndarray:
    """
    Downsample image using PyTorch average pooling (block averaging).

    Parameters
    ----------
    img : np.ndarray
        Input image, shape (Y, X) or (C, Y, X)
    y_scale : int
        Downsampling factor for Y dimension
    x_scale : int
        Downsampling factor for X dimension

    Returns
    -------
    np.ndarray
        Downsampled image with same dtype as input
    """
    if not HAS_TORCH:
        msg = (
            "PyTorch is required for Laplacian pyramid. Install with: pip install torch"
        )
        raise ImportError(msg)

    original_dtype = img.dtype
    is_single_channel = img.ndim == 2

    # Convert to torch tensor
    img_torch = torch.from_numpy(img).float()

    # Add batch dimension if single channel: (Y, X) -> (1, 1, Y, X)
    # Multi-channel: (C, Y, X) -> (1, C, Y, X)
    if is_single_channel:
        img_torch = img_torch.unsqueeze(0).unsqueeze(0)  # (1, 1, Y, X)
    else:
        img_torch = img_torch.unsqueeze(0)  # (1, C, Y, X)

    # Apply average pooling (block averaging)
    downsampled = F.avg_pool2d(
        img_torch,
        kernel_size=(y_scale, x_scale),
        stride=(y_scale, x_scale),
        padding=0,
    )

    # Remove batch dimension and convert back to numpy
    downsampled = downsampled.squeeze(0).numpy()  # (C, Y, X) or (1, Y, X)

    # For single channel, remove the channel dimension if it was added
    if is_single_channel and downsampled.ndim == 3:
        downsampled = downsampled.squeeze(0)  # (Y, X)

    # Convert back to original dtype
    return downsampled.astype(original_dtype)


def _upsample_with_torch(
    img: np.ndarray,
    target_size: tuple[int, int],
    mode: str = "bilinear",
) -> np.ndarray:
    """
    Upsample image using PyTorch interpolation.

    Parameters
    ----------
    img : np.ndarray
        Input image, shape (Y, X) or (C, Y, X)
    target_size : tuple[int, int]
        Target size (Y_target, X_target)
    mode : str
        Interpolation mode: "bilinear" or "bicubic"

    Returns
    -------
    np.ndarray
        Upsampled image with same dtype as input
    """
    if not HAS_TORCH:
        msg = (
            "PyTorch is required for Laplacian pyramid. Install with: pip install torch"
        )
        raise ImportError(msg)

    if mode not in ("bilinear", "bicubic"):
        msg = f"mode must be 'bilinear' or 'bicubic', got {mode}"
        raise ValueError(msg)

    original_dtype = img.dtype
    is_single_channel = img.ndim == 2

    # Convert to torch tensor
    img_torch = torch.from_numpy(img).float()

    # Add batch dimension if single channel: (Y, X) -> (1, 1, Y, X)
    # Multi-channel: (C, Y, X) -> (1, C, Y, X)
    if is_single_channel:
        img_torch = img_torch.unsqueeze(0).unsqueeze(0)  # (1, 1, Y, X)
    else:
        img_torch = img_torch.unsqueeze(0)  # (1, C, Y, X)

    # Upsample using interpolation
    upsampled = F.interpolate(
        img_torch,
        size=target_size,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        antialias=True if mode == "bicubic" else False,
    )

    # Remove batch dimension and convert back to numpy
    upsampled = upsampled.squeeze(0).numpy()  # (C, Y, X) or (1, Y, X)

    # For single channel, remove the channel dimension if it was added
    if is_single_channel and upsampled.ndim == 3:
        upsampled = upsampled.squeeze(0)  # (Y, X)

    # Convert back to original dtype
    return upsampled.astype(original_dtype)


def _load_and_process_image(
    filepath: Path,
    dtype: np.dtype | None,
    normalize: bool = False,
    normalize_mean: float | None = None,
    normalize_std: float | None = None,
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
    normalize : bool
        Whether to normalize the image (per-image mean/std)
    normalize_mean : float | None
        Mean value for normalization (if provided, uses this instead of image mean)
    normalize_std : float | None
        Standard deviation for normalization (if provided, uses this instead of image std)

    Returns
    -------
    np.ndarray
        Processed image array
    """
    img = _load_image(filepath)

    # Normalize to (C, Y, X) if multi-channel
    if img.ndim == 3 and img.shape[2] <= 4:  # (Y, X, C)
        img = np.transpose(img, (2, 0, 1))  # (C, Y, X)

    # Apply normalization
    img = _normalize_image(img, normalize, normalize_mean, normalize_std)

    # Convert dtype if needed
    if dtype is not None and img.dtype != dtype:
        img = img.astype(dtype)

    return img


def _load_image_worker(args: tuple) -> tuple[int, np.ndarray]:
    """
    Worker function for parallel image loading only (no writing).

    This function loads and processes a single image, returning it along with
    its z-index. The actual writing to Zarr happens sequentially in the main process
    to avoid race conditions with compressed chunks.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - z_idx: int - Z-index in the zarr array
        - filepath: Path - Path to image file
        - dtype: np.dtype | None - Target dtype
        - normalize: bool - Whether to normalize
        - normalize_mean: float | None - Mean for normalization
        - normalize_std: float | None - Std for normalization

    Returns
    -------
    tuple[int, np.ndarray]
        (z_idx, processed_image) tuple
    """
    z_idx, filepath, dtype, normalize, normalize_mean, normalize_std = args

    try:
        img = _load_and_process_image(
            filepath, dtype, normalize, normalize_mean, normalize_std
        )
        return (z_idx, img)
    except Exception:
        import traceback

        traceback.print_exc()
        # Return None to indicate failure
        return (z_idx, None)


def _load_and_downsample_worker(args: tuple) -> tuple[int, list[np.ndarray]]:
    """
    Worker function for parallel image loading and downsampling (no writing).

    This function loads an image, downsamples it progressively for all pyramid levels,
    and returns the processed images. The actual writing to Zarr happens sequentially
    in the main process to avoid race conditions with compressed chunks.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - z_idx: int - Z-index in the zarr array
        - filepath: Path - Path to image file
        - pyramid_level_shapes: list[tuple] - Shapes for each pyramid level
        - pyramid_scale_factors: list[tuple] - Cumulative scale factors for each level
        - dtype: np.dtype - Target dtype
        - has_channels: bool - Whether image has channels
        - axis_order: str - Final axis order (e.g., "ZCYX", "CZYX")
        - C: int - Number of channels (for multi-channel)
        - Y: int - Image height
        - X: int - Image width
        - normalize: bool - Whether to normalize
        - normalize_mean: float | None - Mean for normalization
        - normalize_std: float | None - Std for normalization

    Returns
    -------
    tuple[int, list[np.ndarray]]
        (z_idx, list_of_downsampled_images) tuple
        The list contains images for all pyramid levels in order (base level first)
    """
    (
        z_idx,
        filepath,
        pyramid_level_shapes,
        pyramid_scale_factors,
        dtype,
        has_channels,
        axis_order,
        C,
        Y,
        X,
        normalize,
        normalize_mean,
        normalize_std,
    ) = args

    try:
        # Load and process image
        img = _load_and_process_image(
            filepath, dtype, normalize, normalize_mean, normalize_std
        )

        # Apply axis order transformation if needed
        if has_channels:
            # We have img as (C, Y, X), need to prepare for z_idx
            # Create a (1, C, Y, X) array, apply transformation
            slice_data = img[np.newaxis, ...]  # (1, C, Y, X)
            slice_reordered, _ = _apply_axis_order(
                slice_data,
                (1, C, Y, X),
                axis_order,
            )
            img_reordered = slice_reordered[
                0
            ]  # Remove Z dimension, now (C, Y, X) or reordered
        else:
            img_reordered = img  # (Y, X)

        # Start with base level image
        pyramid_images = [img_reordered.copy()]

        # Now downsample progressively for each pyramid level
        # For 2D mode, we downsample Y and X dimensions only
        current_img = img_reordered.copy()
        prev_scale_factors = None

        for _level_idx, (_expected_level_shape, cumulative_scale_factors) in enumerate(
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
                    padded.reshape(
                        C_dim,
                        Y_padded // y_scale_int,
                        y_scale_int,
                        X_padded // x_scale_int,
                        x_scale_int,
                    )
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
                    padded.reshape(
                        Y_padded // y_scale_int,
                        y_scale_int,
                        X_padded // x_scale_int,
                        x_scale_int,
                    )
                    .mean(axis=(1, 3))
                    .astype(dtype)
                )

            pyramid_images.append(downsampled.copy())
            current_img = downsampled
            prev_scale_factors = cumulative_scale_factors

        return (z_idx, pyramid_images)
    except Exception:
        import traceback

        traceback.print_exc()
        # Return None to indicate failure
        return (z_idx, None)


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
        normalize,
        normalize_mean,
        normalize_std,
    ) = args

    try:
        # Load and process image
        img = _load_and_process_image(
            filepath, dtype, normalize, normalize_mean, normalize_std
        )

        # Open zarr group (read-write mode supports concurrent writes)
        # If the group was created with a ProcessSynchronizer, zarr will automatically
        # use it for coordinating concurrent writes, preventing race conditions
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
            img_reordered = slice_reordered[
                0
            ]  # Remove Z dimension, now (C, Y, X) or reordered
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
                    padded.reshape(
                        C_dim,
                        Y_padded // y_scale_int,
                        y_scale_int,
                        X_padded // x_scale_int,
                        x_scale_int,
                    )
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
                    padded.reshape(
                        Y_padded // y_scale_int,
                        y_scale_int,
                        X_padded // x_scale_int,
                        x_scale_int,
                    )
                    .mean(axis=(1, 3))
                    .astype(dtype)
                )

            # Write downsampled image to this pyramid level
            # Get a fresh reference to the array to avoid stale cache issues
            level_array = zarr_group[str(level_idx)]

            # Validate and write - expected_level_shape is from the loop iteration
            if has_channels:
                # downsampled shape is (C, Y, X)
                # We need to match the spatial dimensions (Y, X) from expected_level_shape
                if axis_order == "CZYX":
                    # Array shape: (C, Z, Y, X)
                    # Expected: (C, Z, Y, X) -> slice at z_idx should be (C, Y, X)
                    (
                        expected_C,
                        expected_Z,
                        expected_Y,
                        expected_X,
                    ) = expected_level_shape
                    C_actual, Y_actual, X_actual = downsampled.shape

                    # Fix shape if needed
                    if (
                        C_actual != expected_C
                        or Y_actual != expected_Y
                        or X_actual != expected_X
                    ):
                        if (
                            C_actual > expected_C
                            or Y_actual > expected_Y
                            or X_actual > expected_X
                        ):
                            downsampled = downsampled[
                                :expected_C, :expected_Y, :expected_X
                            ]
                        elif (
                            C_actual < expected_C
                            or Y_actual < expected_Y
                            or X_actual < expected_X
                        ):
                            padded = np.zeros(
                                (expected_C, expected_Y, expected_X),
                                dtype=downsampled.dtype,
                            )
                            padded[:C_actual, :Y_actual, :X_actual] = downsampled
                            downsampled = padded

                    level_array[:, z_idx, :, :] = downsampled
                elif axis_order == "ZCYX":
                    # Array shape: (Z, C, Y, X)
                    # Expected: (Z, C, Y, X) -> slice at z_idx should be (C, Y, X)
                    (
                        expected_Z,
                        expected_C,
                        expected_Y,
                        expected_X,
                    ) = expected_level_shape
                    C_actual, Y_actual, X_actual = downsampled.shape

                    # Fix shape if needed
                    if (
                        C_actual != expected_C
                        or Y_actual != expected_Y
                        or X_actual != expected_X
                    ):
                        if (
                            C_actual > expected_C
                            or Y_actual > expected_Y
                            or X_actual > expected_X
                        ):
                            downsampled = downsampled[
                                :expected_C, :expected_Y, :expected_X
                            ]
                        elif (
                            C_actual < expected_C
                            or Y_actual < expected_Y
                            or X_actual < expected_X
                        ):
                            padded = np.zeros(
                                (expected_C, expected_Y, expected_X),
                                dtype=downsampled.dtype,
                            )
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
                            fixed = np.zeros(
                                expected_slice_shape, dtype=downsampled.dtype
                            )
                            slices = tuple(
                                slice(0, min(d1, d2))
                                for d1, d2 in zip(
                                    downsampled.shape, expected_slice_shape
                                )
                            )
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
                        padded = np.zeros(
                            (expected_Y, expected_X), dtype=downsampled.dtype
                        )
                        padded[:Y_actual, :X_actual] = downsampled
                        downsampled = padded

                # Final verification
                assert (
                    downsampled.shape == (expected_Y, expected_X)
                ), f"Shape fix failed: {downsampled.shape} != ({expected_Y}, {expected_X})"

                # Write with exact shape match
                level_array[z_idx, :, :] = downsampled

            # Update for next level
            current_img = downsampled
            prev_scale_factors = cumulative_scale_factors

        # Explicitly release zarr group reference to help with cleanup on Linux
        # Zarr groups don't have close(), but releasing references helps GC
        del zarr_group
        del base_array
        del level_array
        import gc

        gc.collect()  # Force garbage collection to release file handles

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
        normalize,
        normalize_mean,
        normalize_std,
    ) = args

    try:
        # Load and process image
        img = _load_and_process_image(
            filepath, dtype, normalize, normalize_mean, normalize_std
        )

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
        normalize,
        normalize_mean,
        normalize_std,
    ) = args

    try:
        # Load and process image
        img = _load_and_process_image(
            filepath, dtype, normalize, normalize_mean, normalize_std
        )

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
    normalize: bool = False,
    normalize_mean: float | None = None,
    normalize_std: float | None = None,
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
    normalize : bool
        Whether to normalize images by subtracting mean and dividing by std.
        Normalization is applied only to the full-resolution image.
        Default is False.
    normalize_mean : float | None
        Global mean for normalization across all images. If None and normalize=True,
        mean is computed per image. Default is None.
    normalize_std : float | None
        Global standard deviation for normalization across all images. If None and
        normalize=True, std is computed per image. Default is None.

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

        # Determine chunk size - CRITICAL: Force chunks to match full slice for HPC performance
        # This ensures "one slice = one file", maximizing write bandwidth and minimizing metadata requests
        if zarr_chunks is None:
            if has_channels:
                # Force chunks to match full slice: (1, C, Y, X) or (C, 1, Y, X)
                if final_axis_order == "ZCYX":
                    zarr_chunks = (1, C, Y, X)  # Full slice per chunk
                elif final_axis_order == "CZYX":
                    zarr_chunks = (C, 1, Y, X)  # Full slice per chunk
                else:
                    # Generic: use first dimension as 1, rest as full slice
                    zarr_chunks = (1, *final_shape[1:])
            else:
                # Single channel: full slice per chunk (1, Y, X)
                zarr_chunks = (1, Y, X)

        # Step 3: Zarr Creation (if not dry run)
        if not dry_run:
            # Create zarr array
            # Note: Compression is handled by _create_zarr_array for OME-Zarr
            # For simple zarr arrays, we use default compression (can be customized via zarr_chunks)
            zarr_array = zarr.open(
                str(output_path),
                mode="w",
                shape=final_shape,
                chunks=zarr_chunks,
                dtype=dtype,
            )

            # Handle normalization parameters
            # Initialize global_mean and global_std
            global_mean = normalize_mean
            global_std = normalize_std
            if normalize and (global_mean is None or global_std is None):
                # Compute global mean/std if needed
                all_means = []
                all_stds = []
                for _, filepath in file_list:
                    img = _load_image(filepath)
                    all_means.append(float(np.mean(img)))
                    all_stds.append(float(np.std(img)))
                if global_mean is None:
                    global_mean = float(np.mean(all_means))
                if global_std is None:
                    global_std = float(np.mean(all_stds))
                if global_std == 0:
                    global_std = 1.0

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

            # REFACTORED: Use parallel loading with sequential writing to avoid race conditions
            # This eliminates Read-Modify-Write conflicts on compressed chunks
            if use_multiprocessing and len(file_list) > 10:  # Use for large stacks
                # Prepare tasks for parallel loading only (no writing)
                tasks = []
                for z_idx, (_, filepath) in enumerate(file_list):
                    tasks.append(
                        (
                            z_idx,
                            filepath,
                            dtype,
                            normalize,
                            global_mean,
                            global_std,
                        ),
                    )

                # Parallel loading: workers load images, main process writes sequentially
                with multiprocessing.Pool(processes=workers) as pool:
                    # Load images in parallel using imap for ordered results
                    if tqdm is not None:
                        load_results = list(
                            tqdm(
                                pool.imap(_load_image_worker, tasks),
                                total=len(tasks),
                                desc=f"  Loading {basename}",
                                unit="image",
                            ),
                        )
                    else:
                        load_results = list(pool.imap(_load_image_worker, tasks))

                # Sort by z_idx to ensure correct order (imap preserves order, but be safe)
                load_results.sort(key=lambda x: x[0])

                # Sequential writing: write each loaded image to zarr in order
                # This eliminates all race conditions and decompression errors
                if tqdm is not None:
                    write_iter = tqdm(
                        enumerate(load_results),
                        total=len(load_results),
                        desc=f"  Writing {basename}",
                        unit="image",
                    )
                else:
                    write_iter = enumerate(load_results)

                failures = []
                for _result_idx, (z_idx, img) in write_iter:
                    if img is None:
                        failures.append(z_idx)
                        continue

                    try:
                        # Write to zarr sequentially in main process
                        if has_channels:
                            # Apply axis order transformation for this slice
                            if final_axis_order == "CZYX":
                                zarr_array[:, z_idx, :, :] = img
                            elif final_axis_order == "ZCYX":
                                zarr_array[z_idx, :, :, :] = img
                            else:
                                # For other axis orders, reorder the slice
                                slice_data = img[np.newaxis, ...]  # (1, C, Y, X)
                                slice_reordered, _ = _apply_axis_order(
                                    slice_data,
                                    (1, C, Y, X),
                                    final_axis_order,
                                )
                                if final_axis_order[0] == "Z":
                                    zarr_array[z_idx, ...] = slice_reordered[0]
                                else:
                                    zarr_array[:, z_idx, ...] = slice_reordered[
                                        :, 0, ...
                                    ]
                        else:
                            # Single channel: write directly
                            zarr_array[z_idx, :, :] = img
                    except Exception:
                        failures.append(z_idx)
                        import traceback

                        traceback.print_exc()

                if failures:
                    print(f"Warning: {len(failures)} images failed to write")
            else:
                # Sequential or small stack: load all first, then write
                if use_multiprocessing and len(file_list) > 1:
                    # Parallel loading only
                    load_func = partial(
                        _load_and_process_image,
                        dtype=dtype,
                        normalize=normalize,
                        normalize_mean=global_mean,
                        normalize_std=global_std,
                    )
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
                        _load_and_process_image(
                            filepath,
                            dtype=dtype,
                            normalize=normalize,
                            normalize_mean=global_mean,
                            normalize_std=global_std,
                        )
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
                            _load_and_process_image(
                                filepath,
                                dtype=dtype,
                                normalize=normalize,
                                normalize_mean=global_mean,
                                normalize_std=global_std,
                            ),
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
    normalize: bool = False,
    normalize_mean: float | None = None,
    normalize_std: float | None = None,
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
    normalize : bool
        Whether to normalize images. Default: False
        If True, applies mean subtraction and division by standard deviation.
        - If normalize_mean and normalize_std are None: per-image normalization
        - If normalize_mean and normalize_std are provided: global normalization across all images
    normalize_mean : float | None
        Mean value for normalization. If None and normalize=True, uses per-image mean.
        If provided, uses this value for all images (global normalization).
    normalize_std : float | None
        Standard deviation for normalization. If None and normalize=True, uses per-image std.
        If provided, uses this value for all images (global normalization).
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
        print(
            f"File pattern: {pattern.pattern if isinstance(pattern, re.Pattern) else pattern}"
        )
        print(f"Extension: {extension}")

    # Step 2: Stack Analysis (reuse logic from stack_files_to_zarr)
    results = {}

    for stack_idx, (basename, file_list) in enumerate(stacks.items(), 1):
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"[{stack_idx}/{len(stacks)}] Processing stack: {basename}")
            print(f"  Files found: {len(file_list)}")
            print(
                f"  Counter range: {min(c for c, _ in file_list)} - {max(c for c, _ in file_list)}"
            )
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

        # Compute global mean/std if normalize=True and global normalization requested
        # (normalize_mean and normalize_std are provided)
        global_mean = normalize_mean
        global_std = normalize_std
        if normalize and (normalize_mean is not None or normalize_std is not None):
            # If only one is provided, compute the other across all images
            if normalize_mean is None or normalize_std is None:
                if verbose:
                    print("  Computing global statistics across all images...")
                all_means = []
                all_stds = []
                for _, filepath in file_list:
                    img = _load_image(filepath)
                    # Normalize to (C, Y, X) if multi-channel
                    if img.ndim == 3 and img.shape[2] <= 4:  # (Y, X, C)
                        img = np.transpose(img, (2, 0, 1))  # (C, Y, X)
                    all_means.append(float(np.mean(img)))
                    all_stds.append(float(np.std(img)))
                if normalize_mean is None:
                    global_mean = float(np.mean(all_means))
                if normalize_std is None:
                    global_std = float(
                        np.mean(all_stds)
                    )  # Use mean of stds, or could use pooled std
                if verbose:
                    print(
                        f"  Global mean: {global_mean:.6f}, Global std: {global_std:.6f}"
                    )

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
        else:
            raise ValueError(
                f"Invalid downsample_mode: {downsample_mode}. Must be '2d'."
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

            if verbose:
                print(f"  Creating OME-Zarr: {output_path}", flush=True)
                print(f"  Base shape: {base_shape}, dtype: {dtype}", flush=True)
                print(f"  Pyramid levels: {num_pyramid_levels}", flush=True)
                print(f"  Downsample method: {downsample_method}", flush=True)
                print(f"  Downsample mode: {downsample_mode}", flush=True)
                print(
                    "\n  *** STARTING PROCESSING - THIS MAY TAKE A WHILE ***",
                    flush=True,
                )
                print("  *** WATCH FOR PROGRESS BARS BELOW ***\n", flush=True)

            # Determine if we'll use multiprocessing (needed for synchronizer)
            import multiprocessing

            if num_workers is None:
                will_use_multiprocessing = multiprocessing.cpu_count() > 1
            elif num_workers > 1:
                will_use_multiprocessing = True
            else:
                will_use_multiprocessing = False

            # Create OME-Zarr root group
            # Use ProcessSynchronizer for concurrent writes when using multiprocessing
            # Note: Disable on Linux with Python < 3.11 due to hanging issues with file locks
            import sys

            is_linux_python_old = sys.platform.startswith(
                "linux"
            ) and sys.version_info < (3, 11)
            use_synchronizer = (
                will_use_multiprocessing
                and ProcessSynchronizer is not None
                and not is_linux_python_old
            )
            if use_synchronizer:
                # Create synchronizer file in the zarr directory
                sync_path = str(output_path / ".zarr_sync")
                synchronizer = ProcessSynchronizer(sync_path)
                if verbose:
                    print(
                        "  Creating zarr root group with ProcessSynchronizer for concurrent writes...",
                        flush=True,
                    )
                root = zarr.open_group(
                    str(output_path), mode="w", synchronizer=synchronizer
                )
            else:
                if verbose:
                    if will_use_multiprocessing:
                        if ProcessSynchronizer is None:
                            print(
                                "  Creating zarr root group (ProcessSynchronizer not available, using default)...",
                                flush=True,
                            )
                            print(
                                "  WARNING: ProcessSynchronizer not available. For concurrent writes, consider installing:",
                                flush=True,
                            )
                            print(
                                "    pip install fasteners  # Required for ProcessSynchronizer",
                                flush=True,
                            )
                        elif is_linux_python_old:
                            print(
                                "  Creating zarr root group (ProcessSynchronizer disabled on Linux/Python < 3.11 to prevent hanging)...",
                                flush=True,
                            )
                        else:
                            print("  Creating zarr root group...", flush=True)
                    else:
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
                            for curr, prev in zip(
                                cumulative_scale_factors, prev_cumulative_scale_factors
                            )
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
                            level_shape = tuple(current_simulated_shape[:-2]) + (
                                Y_new,
                                X_new,
                            )
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
                print(
                    f"  ✓ Calculated {len(pyramid_level_shapes)} pyramid level shapes",
                    flush=True,
                )
                for idx, shape in enumerate(pyramid_level_shapes):
                    print(f"    Level {idx}: {shape}", flush=True)

            # Determine chunk size for all levels - CRITICAL: Force chunks to match full slice
            # Level 0: Full slice per chunk (1, Height, Width) for maximum write bandwidth
            # Pyramid levels: Spatial-only chunks, avoid spanning entire Z-depth
            if zarr_chunks is None:
                if has_channels:
                    if final_axis_order == "ZCYX":
                        # Full slice per chunk: (1, C, Y, X)
                        base_chunks = (1, C, Y, X)
                    elif final_axis_order == "CZYX":
                        # Full slice per chunk: (C, 1, Y, X)
                        base_chunks = (C, 1, Y, X)
                    else:
                        # Generic: full slice per chunk
                        base_chunks = (1,) + tuple(base_shape[1:])
                else:
                    # Single channel: full slice per chunk (1, Y, X)
                    base_chunks = (1, Y, X)
            else:
                base_chunks = zarr_chunks

            # Create ALL pyramid level arrays upfront (empty, we'll write to them in parallel)
            # Use zarr 3.0+ API: shape must be a keyword argument
            if verbose:
                print(
                    "\n    Creating all pyramid level zarr arrays (empty, will write in parallel)...",
                    flush=True,
                )
                print(
                    f"    Creating base level (0) with shape {base_shape}...",
                    flush=True,
                )
            base_zarr_array = _create_zarr_array(
                root,
                "0",
                shape=base_shape,
                chunks=base_chunks,
                dtype=dtype,
            )
            pyramid_zarr_arrays = [base_zarr_array]
            if verbose:
                print("    ✓ Created base level (0)", flush=True)

            # Create pyramid level arrays
            for level_idx, level_shape in enumerate(pyramid_level_shapes[1:], start=1):
                if verbose:
                    print(
                        f"    Creating pyramid level {level_idx} with shape {level_shape}...",
                        flush=True,
                    )
                # Pyramid levels: Ensure chunks are spatial-only or incremental
                # Avoid chunks that span entire Z-depth during write phase
                if has_channels:
                    if final_axis_order == "ZCYX":
                        # Full slice per chunk: (1, C, Y_level, X_level)
                        level_chunks = (
                            1,
                            level_shape[1],
                            level_shape[2],
                            level_shape[3],
                        )
                    elif final_axis_order == "CZYX":
                        # Full slice per chunk: (C, 1, Y_level, X_level)
                        level_chunks = (
                            level_shape[0],
                            1,
                            level_shape[2],
                            level_shape[3],
                        )
                    else:
                        # Generic: full slice per chunk
                        level_chunks = (1,) + tuple(level_shape[1:])
                else:
                    # Single channel: full slice per chunk (1, Y_level, X_level)
                    level_chunks = (1, level_shape[1], level_shape[2])
                # Zarr 3.0+ API: shape must be a keyword argument
                level_array = _create_zarr_array(
                    root,
                    str(level_idx),
                    shape=level_shape,
                    chunks=level_chunks,
                    dtype=dtype,
                )
                pyramid_zarr_arrays.append(level_array)
                if verbose:
                    print(f"    ✓ Created pyramid level {level_idx}", flush=True)

            if verbose:
                print(
                    f"    ✓ Created {len(pyramid_level_shapes)} pyramid level arrays",
                    flush=True,
                )
                print("\n" + "=" * 70, flush=True)
                print(
                    "  [STEP 1/1] LOADING + DOWNSAMPLING + WRITING TO ALL PYRAMID LEVELS - ULTRA FAST MODE",
                    flush=True,
                )
                print("=" * 70 + "\n", flush=True)
                import sys

                sys.stdout.flush()
                sys.stderr.flush()

            # Setup multiprocessing (already determined above, but keep for consistency)
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
                    print(
                        f"    Using multiprocessing with {workers} workers (one per core)",
                        flush=True,
                    )
                    print(
                        f"    Processing {len(file_list)} images: load → downsample → write to all {num_pyramid_levels} levels",
                        flush=True,
                    )
                else:
                    print("    Using sequential processing (1 worker)", flush=True)
                    print(f"    Processing {len(file_list)} images...", flush=True)

            # REFACTORED: Use parallel loading/downsampling with sequential writing
            # This eliminates Read-Modify-Write conflicts on compressed chunks
            if use_multiprocessing and len(file_list) > 10:
                # Prepare tasks for parallel loading and downsampling only (no writing)
                tasks = []
                for z_idx, (_, filepath) in enumerate(file_list):
                    tasks.append(
                        (
                            z_idx,
                            filepath,
                            pyramid_level_shapes,  # All pyramid level shapes
                            pyramid_scale_factors,  # Cumulative scale factors
                            dtype,
                            has_channels,
                            final_axis_order,
                            C,
                            Y,
                            X,
                            normalize,
                            global_mean,
                            global_std,
                        ),
                    )
                if verbose:
                    print(
                        f"\n    Starting multiprocessing pool with {workers} workers for parallel loading/downsampling...",
                        flush=True,
                    )
                    print(
                        "    Images will be loaded and downsampled in parallel, then written sequentially to avoid race conditions",
                        flush=True,
                    )

                # Parallel loading and downsampling: workers process images, main process writes sequentially
                import sys

                if sys.platform.startswith("linux") and sys.version_info < (3, 11):
                    ctx = multiprocessing.get_context("spawn")
                    pool = ctx.Pool(processes=workers)
                else:
                    pool = multiprocessing.Pool(processes=workers)

                try:
                    # Load and downsample images in parallel using imap for ordered results
                    if tqdm is not None:
                        if verbose:
                            print("", flush=True)  # Blank line before progress bar
                        load_results = list(
                            tqdm(
                                pool.imap(_load_and_downsample_worker, tasks),
                                total=len(tasks),
                                desc="    LOAD+DOWNSAMPLE",
                                unit="img",
                                ncols=100,
                                miniters=1,
                            )
                        )
                        if verbose:
                            print("", flush=True)  # Blank line after progress bar
                    else:
                        load_results = list(
                            pool.imap(_load_and_downsample_worker, tasks)
                        )

                    # Sort by z_idx to ensure correct order (imap preserves order, but be safe)
                    load_results.sort(key=lambda x: x[0])

                    # Sequential writing: write each loaded/downsampled image set to zarr in order
                    # This eliminates all race conditions and decompression errors
                    if tqdm is not None:
                        if verbose:
                            print("", flush=True)  # Blank line before write progress
                        write_iter = tqdm(
                            enumerate(load_results),
                            total=len(load_results),
                            desc="    WRITE TO ZARR",
                            unit="img",
                            ncols=100,
                            miniters=1,
                        )
                    else:
                        write_iter = enumerate(load_results)

                    failures = []
                    for _result_idx, (z_idx, pyramid_images) in write_iter:
                        if pyramid_images is None:
                            failures.append(z_idx)
                            continue

                        try:
                            # Write to all pyramid levels sequentially in main process
                            for level_idx, level_img in enumerate(pyramid_images):
                                level_array = pyramid_zarr_arrays[level_idx]

                                if has_channels:
                                    if final_axis_order == "CZYX":
                                        level_array[:, z_idx, :, :] = level_img
                                    elif final_axis_order == "ZCYX":
                                        level_array[z_idx, :, :, :] = level_img
                                    else:
                                        # Generic: assume Z is first dimension
                                        level_array[z_idx, ...] = level_img
                                else:
                                    # Single channel: write directly
                                    level_array[z_idx, :, :] = level_img
                        except Exception:
                            failures.append(z_idx)
                            import traceback

                            traceback.print_exc()

                    if failures:
                        if verbose:
                            print(
                                f"    WARNING: {len(failures)} images failed to write",
                                flush=True,
                            )

                    if verbose:
                        print(
                            f"\n    ✓ Processed {len(load_results) - len(failures)} images (loaded/downsampled in parallel, written sequentially)",
                            flush=True,
                        )
                finally:
                    pool.close()
                    pool.join()
            else:
                # Sequential writing (small stacks or num_workers=1)
                if verbose:
                    print(
                        "\n    Processing images sequentially with immediate downsampling...",
                        flush=True,
                    )
                    print(
                        f"    PROCESSING {len(file_list)} IMAGES - PROGRESS BAR BELOW:",
                        flush=True,
                    )
                    print("-" * 70, flush=True)

                # Write directly to zarr sequentially with immediate downsampling
                if tqdm is not None:
                    if verbose:
                        print("", flush=True)  # Blank line before progress bar
                    for z_idx, (_, filepath) in enumerate(
                        tqdm(
                            file_list,
                            desc="    LOAD+DOWNSAMPLE+WRITE",
                            unit="img",
                            ncols=100,
                            miniters=1,
                        )
                    ):
                        result = _load_and_write_to_all_pyramid_levels(
                            (
                                z_idx,
                                filepath,
                                str(output_path),
                                pyramid_level_shapes,
                                pyramid_scale_factors,
                                dtype,
                                has_channels,
                                final_axis_order,
                                C,
                                Y,
                                X,
                                normalize,
                                global_mean,
                                global_std,
                            )
                        )
                        if not result[1] and verbose:
                            print(
                                f"    WARNING: Failed to write image {z_idx}",
                                flush=True,
                            )
                    if verbose:
                        print("", flush=True)  # Blank line after progress bar
                else:
                    # Manual progress bar when tqdm not available
                    if verbose:
                        print(
                            "    Processing images with immediate downsampling...",
                            flush=True,
                        )
                        print(f"    [{' ' * 50}] 0%", end="", flush=True)
                    total = len(file_list)
                    completed = 0
                    for _idx, (z_idx, (_, filepath)) in enumerate(
                        enumerate(file_list), 1
                    ):
                        result = _load_and_write_to_all_pyramid_levels(
                            (
                                z_idx,
                                filepath,
                                str(output_path),
                                pyramid_level_shapes,
                                pyramid_scale_factors,
                                dtype,
                                has_channels,
                                final_axis_order,
                                C,
                                Y,
                                X,
                                normalize,
                                global_mean,
                                global_std,
                            )
                        )
                        completed += 1
                        if verbose:
                            percent = 100 * completed // total
                            filled = int(50 * completed / total)
                            bar = "=" * filled + " " * (50 - filled)
                            print(
                                f"\r    [{bar}] {percent}% ({completed}/{total})",
                                end="",
                                flush=True,
                            )
                    if verbose:
                        print("", flush=True)  # New line after progress

                if verbose:
                    print(
                        f"    ✓ Processed {len(file_list)} images with immediate downsampling",
                        flush=True,
                    )

            # All pyramid levels are now written with immediate downsampling!
            # Build metadata for all levels
            if verbose:
                print(
                    f"\n    ✓ All {num_pyramid_levels} pyramid levels written with immediate downsampling!",
                    flush=True,
                )
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
            for level_idx, cumulative_scale_factors in enumerate(
                pyramid_scale_factors, start=1
            ):
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
                print(f"{'=' * 70}", flush=True)
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
        print(f"\n{'=' * 70}")
        print(f"✓ Successfully processed {len(results)} stack(s) as OME-Zarr")
        for stack_name, metadata in results.items():
            print(f"  - {stack_name}: {metadata['zarr_path']}")
            print(
                f"    Shape: {metadata['shape']}, Levels: {metadata['pyramid_levels']}"
            )
    return results


def _load_and_write_laplacian_pyramid(
    args: tuple,
) -> tuple[int, bool]:
    """
    Load an image, build Laplacian pyramid, and write difference maps to Zarr.

    This function builds a Gaussian pyramid (downsampled versions) and then
    computes Laplacian pyramid (difference maps) by upsampling lower levels
    and subtracting from higher levels.

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
        - interpolation_mode: str - "bilinear" or "bicubic"
        - store_base_level: bool - Whether to store lowest resolution level

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
        interpolation_mode,
        store_base_level,
        normalize,
        normalize_mean,
        normalize_std,
    ) = args

    try:
        if not HAS_TORCH:
            msg = "PyTorch is required for Laplacian pyramid. Install with: pip install torch"
            raise ImportError(msg)

        # Load and process image
        img = _load_and_process_image(
            filepath, dtype, normalize, normalize_mean, normalize_std
        )

        # Open zarr group
        zarr_group = zarr.open_group(zarr_group_path, mode="r+")

        # Apply axis order transformation if needed
        if has_channels:
            slice_data = img[np.newaxis, ...]  # (1, C, Y, X)
            slice_reordered, _ = _apply_axis_order(
                slice_data,
                (1, C, Y, X),
                axis_order,
            )
            img_reordered = slice_reordered[0]  # (C, Y, X)
        else:
            img_reordered = img  # (Y, X)

        # Build Gaussian pyramid (downsampled versions)
        gaussian_pyramid = []
        current_img = img_reordered.copy()
        prev_scale_factors = None

        # Add base level (full resolution) to Gaussian pyramid
        gaussian_pyramid.append(current_img)

        # Downsample progressively to build Gaussian pyramid
        for _level_idx, (_expected_level_shape, cumulative_scale_factors) in enumerate(
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

            # Extract Y, X scale factors
            if has_channels:
                y_scale, x_scale = incremental_scale_factors[-2:]
            else:
                y_scale, x_scale = incremental_scale_factors[-2:]

            y_scale_int = int(y_scale)
            x_scale_int = int(x_scale)

            # Pad if needed for downsampling
            if has_channels:
                C_dim, Y_dim, X_dim = current_img.shape
                pad_Y = (y_scale_int - (Y_dim % y_scale_int)) % y_scale_int
                pad_X = (x_scale_int - (X_dim % x_scale_int)) % x_scale_int

                if pad_Y > 0 or pad_X > 0:
                    padded = np.pad(
                        current_img,
                        ((0, 0), (0, pad_Y), (0, pad_X)),
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    padded = current_img
            else:
                Y_dim, X_dim = current_img.shape
                pad_Y = (y_scale_int - (Y_dim % y_scale_int)) % y_scale_int
                pad_X = (x_scale_int - (X_dim % x_scale_int)) % x_scale_int

                if pad_Y > 0 or pad_X > 0:
                    padded = np.pad(
                        current_img,
                        ((0, pad_Y), (0, pad_X)),
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    padded = current_img

            # Downsample using PyTorch
            downsampled = _downsample_with_torch(padded, y_scale_int, x_scale_int)
            gaussian_pyramid.append(downsampled)
            current_img = downsampled
            prev_scale_factors = cumulative_scale_factors

        # Build Laplacian pyramid (difference maps)
        # Start from lowest resolution and work up
        num_levels = len(gaussian_pyramid)

        # Store base level (lowest resolution) if requested
        # Store at highest level number to match standard convention (level 0 = highest resolution)
        if store_base_level:
            base_level = gaussian_pyramid[-1]  # Lowest resolution
            base_level_num = num_levels - 1  # Highest level number
            base_array = zarr_group[str(base_level_num)]
            if has_channels:
                if axis_order == "CZYX":
                    base_array[:, z_idx, :, :] = base_level
                elif axis_order == "ZCYX":
                    base_array[z_idx, :, :, :] = base_level
                else:
                    base_array[z_idx, ...] = base_level
            else:
                base_array[z_idx, :, :] = base_level

        # Compute and store difference maps (from lowest to highest)
        for level_idx in range(num_levels - 1, 0, -1):  # Reverse order
            current_level = gaussian_pyramid[level_idx - 1]  # Higher resolution
            lower_level = gaussian_pyramid[level_idx]  # Lower resolution

            # Upsample lower level to match current level's shape
            target_size = current_level.shape[-2:]  # (Y, X)
            upsampled_lower = _upsample_with_torch(
                lower_level,
                target_size,
                mode=interpolation_mode,
            )

            # Compute difference map
            difference = current_level.astype(np.float32) - upsampled_lower.astype(
                np.float32
            )

            # Crop if needed to match expected shape
            if has_channels:
                expected_shape = pyramid_level_shapes[level_idx - 1]
                if axis_order == "CZYX":
                    expected_C, expected_Z, expected_Y, expected_X = expected_shape
                    expected_slice_shape = (expected_C, expected_Y, expected_X)
                elif axis_order == "ZCYX":
                    expected_Z, expected_C, expected_Y, expected_X = expected_shape
                    expected_slice_shape = (expected_C, expected_Y, expected_X)
                else:
                    expected_slice_shape = expected_shape[1:]  # Skip Z

                if difference.shape != expected_slice_shape:
                    # Crop or pad to match
                    if all(
                        d <= e for d, e in zip(difference.shape, expected_slice_shape)
                    ):
                        # Pad
                        padded_diff = np.zeros(
                            expected_slice_shape, dtype=difference.dtype
                        )
                        slices = tuple(
                            slice(0, min(d, e))
                            for d, e in zip(difference.shape, expected_slice_shape)
                        )
                        padded_diff[slices] = difference[slices]
                        difference = padded_diff
                    else:
                        # Crop
                        slices = tuple(
                            slice(0, min(d, e))
                            for d, e in zip(difference.shape, expected_slice_shape)
                        )
                        difference = difference[slices]
            else:
                expected_shape = pyramid_level_shapes[level_idx - 1]
                expected_Z, expected_Y, expected_X = expected_shape
                expected_slice_shape = (expected_Y, expected_X)

                if difference.shape != expected_slice_shape:
                    if all(
                        d <= e for d, e in zip(difference.shape, expected_slice_shape)
                    ):
                        # Pad
                        padded_diff = np.zeros(
                            expected_slice_shape, dtype=difference.dtype
                        )
                        slices = tuple(
                            slice(0, min(d, e))
                            for d, e in zip(difference.shape, expected_slice_shape)
                        )
                        padded_diff[slices] = difference[slices]
                        difference = padded_diff
                    else:
                        # Crop
                        slices = tuple(
                            slice(0, min(d, e))
                            for d, e in zip(difference.shape, expected_slice_shape)
                        )
                        difference = difference[slices]

            # Convert difference to target dtype
            difference = difference.astype(dtype)

            # Write difference map to Zarr
            # Use "diff_{level_idx}" naming for difference maps
            diff_array_name = f"diff_{level_idx - 1}"
            if diff_array_name not in zarr_group:
                # This shouldn't happen if arrays are pre-created, but handle gracefully
                continue

            diff_array = zarr_group[diff_array_name]
            if has_channels:
                if axis_order == "CZYX":
                    diff_array[:, z_idx, :, :] = difference
                elif axis_order == "ZCYX":
                    diff_array[z_idx, :, :, :] = difference
                else:
                    diff_array[z_idx, ...] = difference
            else:
                diff_array[z_idx, :, :] = difference

        return (z_idx, True)
    except Exception:
        import traceback

        traceback.print_exc()
        return (z_idx, False)


def stack_files_to_ome_zarr_laplacian(
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
    interpolation_mode: str = "bilinear",
    store_base_level: bool = True,
    normalize: bool = False,
    normalize_mean: float | None = None,
    normalize_std: float | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Scan directory for image files and save as OME-Zarr with Laplacian pyramid.

    Creates OME-Zarr format files with Laplacian pyramid (difference maps) instead of
    Gaussian pyramid (downsampled images). This enables perfect reconstruction from
    the lowest resolution plus all difference maps.

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
        Note: Difference maps may contain negative values, so signed types are recommended.
    axis_order : str
        Axis order for multi-channel images. Default: "ZCYX"
    output_naming : Callable[[str], str] | None
        Function to generate output zarr filename from basename.
        If None, uses default: f"{basename}.ome.zarr"
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
    pyramid_scale_factors : list[tuple[int, ...]] | None
        Custom scale factors for each pyramid level (excluding base level).
        Each tuple specifies scale factors for each dimension (Z, C, Y, X).
        If None, uses automatic 2x downsampling per level.
    downsample_mode : str
        Downsampling mode for pyramid generation. Default: "2d"
        - "2d": For 2D operations on 3D grid - downsample only Y, X (not Z)
    downsample_axes : tuple[str, ...] | None
        Explicit control over which axes to downsample. If None, auto-determined from downsample_mode.
    interpolation_mode : str
        Interpolation mode for upsampling: "bilinear" or "bicubic" (default: "bilinear")
    store_base_level : bool
        Whether to store the lowest resolution level (base level). Default: True
        If False, only difference maps are stored (requires all maps for reconstruction).
    normalize : bool
        Whether to normalize images. Default: False
        If True, applies mean subtraction and division by standard deviation.
        - If normalize_mean and normalize_std are None: per-image normalization
        - If normalize_mean and normalize_std are provided: global normalization across all images
    normalize_mean : float | None
        Mean value for normalization. If None and normalize=True, uses per-image mean.
        If provided, uses this value for all images (global normalization).
    normalize_std : float | None
        Standard deviation for normalization. If None and normalize=True, uses per-image std.
        If provided, uses this value for all images (global normalization).
    verbose : bool
        Whether to print detailed progress information. Default: True

    Returns
    -------
    dict[str, dict]
        Dictionary mapping stack basename to metadata:
        {
            "stack_name": {
                "zarr_path": str,
                "shape": tuple[int, ...],  # Full resolution shape
                "dtype": np.dtype,
                "file_count": int,
                "files": list[str],
                "counter_range": tuple[int, int],
                "axis_order": str,
                "pyramid_levels": int,
            }
        }

    Examples
    --------
    >>> from qlty.utils.stack_to_zarr import stack_files_to_ome_zarr_laplacian
    >>> result = stack_files_to_ome_zarr_laplacian(
    ...     directory="/path/to/images",
    ...     extension=".tif",
    ...     pattern=r"(.+)_(\\d+)\\.tif$",
    ...     pyramid_levels=4,
    ...     interpolation_mode="bilinear",
    ... )
    """
    if not HAS_TORCH:
        msg = (
            "PyTorch is required for Laplacian pyramid. Install with: pip install torch"
        )
        raise ImportError(msg)

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

    # Step 1: File Discovery and Parsing (same as stack_files_to_ome_zarr)
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
        print(
            f"File pattern: {pattern.pattern if isinstance(pattern, re.Pattern) else pattern}"
        )
        print(f"Extension: {extension}")

    # Step 2: Stack Analysis (same as stack_files_to_ome_zarr)
    results = {}

    for stack_idx, (basename, file_list) in enumerate(stacks.items(), 1):
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"[{stack_idx}/{len(stacks)}] Processing stack: {basename}")
            print(f"  Files found: {len(file_list)}")
            print(
                f"  Counter range: {min(c for c, _ in file_list)} - {max(c for c, _ in file_list)}"
            )
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

        # For difference maps, we may need signed types if values can be negative
        # Use float32 for difference maps to handle negative values
        diff_dtype = np.float32

        # Compute global mean/std if normalize=True and global normalization requested
        # (normalize_mean and normalize_std are provided)
        global_mean = normalize_mean
        global_std = normalize_std
        if normalize and (normalize_mean is not None or normalize_std is not None):
            # If only one is provided, compute the other across all images
            if normalize_mean is None or normalize_std is None:
                if verbose:
                    print("  Computing global statistics across all images...")
                all_means = []
                all_stds = []
                for _, filepath in file_list:
                    img = _load_image(filepath)
                    # Normalize to (C, Y, X) if multi-channel
                    if img.ndim == 3 and img.shape[2] <= 4:  # (Y, X, C)
                        img = np.transpose(img, (2, 0, 1))  # (C, Y, X)
                    all_means.append(float(np.mean(img)))
                    all_stds.append(float(np.std(img)))
                if normalize_mean is None:
                    global_mean = float(np.mean(all_means))
                if normalize_std is None:
                    global_std = float(
                        np.mean(all_stds)
                    )  # Use mean of stds, or could use pooled std
                if verbose:
                    print(
                        f"  Global mean: {global_mean:.6f}, Global std: {global_std:.6f}"
                    )

        if verbose:
            print(f"  Image dimensions: {first_image.shape}")
            print(f"  Detected shape: {base_shape}")
            print(f"  Data type: {dtype}")
            print(f"  Difference map dtype: {diff_dtype}")
            print(f"  Axis order: {final_axis_order}")
            if has_channels:
                print(f"  Channels: {C}")

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
                            f"expected ({C}, {Y, X})"
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

        # Determine pyramid levels and scale factors (same logic as stack_files_to_ome_zarr)
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
        else:
            raise ValueError(
                f"Invalid downsample_mode: {downsample_mode}. Must be '2d'."
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
                print(
                    f"  Creating Laplacian pyramid OME-Zarr: {output_path}", flush=True
                )
                print(f"  Base shape: {base_shape}, dtype: {dtype}", flush=True)
                print(f"  Pyramid levels: {num_pyramid_levels}", flush=True)
                print(f"  Interpolation mode: {interpolation_mode}", flush=True)
                print(f"  Store base level: {store_base_level}", flush=True)
                print(
                    "\n  *** STARTING PROCESSING - THIS MAY TAKE A WHILE ***",
                    flush=True,
                )
                print("  *** WATCH FOR PROGRESS BARS BELOW ***\n", flush=True)

            # Determine if we'll use multiprocessing
            import multiprocessing

            if num_workers is None:
                will_use_multiprocessing = multiprocessing.cpu_count() > 1
            elif num_workers > 1:
                will_use_multiprocessing = True
            else:
                will_use_multiprocessing = False

            # Create OME-Zarr root group
            import sys

            is_linux_python_old = sys.platform.startswith(
                "linux"
            ) and sys.version_info < (3, 11)
            use_synchronizer = (
                will_use_multiprocessing
                and ProcessSynchronizer is not None
                and not is_linux_python_old
            )
            if use_synchronizer:
                sync_path = str(output_path / ".zarr_sync")
                synchronizer = ProcessSynchronizer(sync_path)
                if verbose:
                    print(
                        "  Creating zarr root group with ProcessSynchronizer for concurrent writes...",
                        flush=True,
                    )
                root = zarr.open_group(
                    str(output_path), mode="w", synchronizer=synchronizer
                )
            else:
                if verbose:
                    if will_use_multiprocessing:
                        if ProcessSynchronizer is None:
                            print(
                                "  Creating zarr root group (ProcessSynchronizer not available, using default)...",
                                flush=True,
                            )
                        elif is_linux_python_old:
                            print(
                                "  Creating zarr root group (ProcessSynchronizer disabled on Linux/Python < 3.11)...",
                                flush=True,
                            )
                        else:
                            print("  Creating zarr root group...", flush=True)
                    else:
                        print("  Creating zarr root group...", flush=True)
                root = zarr.open_group(str(output_path), mode="w")

            if verbose:
                print("  ✓ Zarr root group created", flush=True)
                print("  Calculating pyramid level shapes...", flush=True)

            # Calculate pyramid level shapes progressively (same as stack_files_to_ome_zarr)
            pyramid_level_shapes = [base_shape]
            if num_pyramid_levels > 1:
                current_simulated_shape = list(base_shape)
                prev_cumulative_scale_factors = None

                for cumulative_scale_factors in pyramid_scale_factors:
                    if prev_cumulative_scale_factors is None:
                        incremental_scale_factors = cumulative_scale_factors
                    else:
                        incremental_scale_factors = tuple(
                            curr / prev if prev > 0 else curr
                            for curr, prev in zip(
                                cumulative_scale_factors, prev_cumulative_scale_factors
                            )
                        )

                    # Extract Y, X scale factors for 2D downsampling
                    if has_channels:
                        y_scale = incremental_scale_factors[-2]
                        x_scale = incremental_scale_factors[-1]
                        if final_axis_order == "ZCYX":
                            Y_dim = current_simulated_shape[2]
                            X_dim = current_simulated_shape[3]
                        elif final_axis_order == "CZYX":
                            Y_dim = current_simulated_shape[2]
                            X_dim = current_simulated_shape[3]
                        else:
                            Y_dim = current_simulated_shape[-2]
                            X_dim = current_simulated_shape[-1]
                    else:
                        y_scale, x_scale = incremental_scale_factors[-2:]
                        Y_dim = current_simulated_shape[1]
                        X_dim = current_simulated_shape[2]

                    # Calculate padding
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
                            level_shape = tuple(current_simulated_shape[:-2]) + (
                                Y_new,
                                X_new,
                            )
                    else:
                        level_shape = (
                            current_simulated_shape[0],  # Z unchanged
                            Y_new,
                            X_new,
                        )

                    pyramid_level_shapes.append(level_shape)
                    current_simulated_shape = list(level_shape)
                    prev_cumulative_scale_factors = cumulative_scale_factors

            if verbose:
                print(
                    f"  ✓ Calculated {len(pyramid_level_shapes)} pyramid level shapes",
                    flush=True,
                )
                for idx, shape in enumerate(pyramid_level_shapes):
                    print(f"    Level {idx}: {shape}", flush=True)

            # Determine chunk size
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

            # Create Zarr arrays for Laplacian pyramid
            # Base level (lowest resolution) if store_base_level=True
            # Difference maps for each level (diff_0, diff_1, etc.)
            if verbose:
                print(
                    "\n    Creating Laplacian pyramid zarr arrays...",
                    flush=True,
                )

            # Create base level array (lowest resolution)
            # Store at highest level number to match standard convention (level 0 = highest resolution)
            if store_base_level:
                base_level_shape = pyramid_level_shapes[-1]  # Lowest resolution
                base_level_num = num_pyramid_levels - 1  # Highest level number
                if verbose:
                    print(
                        f"    Creating base level ({base_level_num}) with shape {base_level_shape}...",
                        flush=True,
                    )
                _create_zarr_array(
                    root,
                    str(base_level_num),
                    shape=base_level_shape,
                    chunks=base_chunks,
                    dtype=dtype,
                )
                if verbose:
                    print(f"    ✓ Created base level ({base_level_num})", flush=True)

            # Create difference map arrays
            # Following standard convention: level 0 = highest resolution
            # diff_0 corresponds to difference between level 0 (full res) and level 1
            # diff_1 corresponds to difference between level 1 and level 2
            # etc.
            # Base level (lowest resolution) is stored at highest level number
            # So we need num_pyramid_levels - 1 difference maps
            for diff_idx in range(num_pyramid_levels - 1):
                # Difference map shape matches the higher resolution level
                diff_shape = pyramid_level_shapes[diff_idx]
                diff_name = f"diff_{diff_idx}"

                # Determine chunks for difference map
                if has_channels:
                    if final_axis_order == "ZCYX":
                        diff_chunks = (
                            1,
                            min(diff_shape[1], 4),
                            min(diff_shape[2], 256),
                            min(diff_shape[3], 256),
                        )
                    elif final_axis_order == "CZYX":
                        diff_chunks = (
                            min(diff_shape[0], 4),
                            1,
                            min(diff_shape[2], 256),
                            min(diff_shape[3], 256),
                        )
                    else:
                        diff_chunks = (1,) + tuple(min(d, 256) for d in diff_shape[1:])
                else:
                    diff_chunks = (
                        1,
                        min(diff_shape[1], 256),
                        min(diff_shape[2], 256),
                    )

                if verbose:
                    print(
                        f"    Creating difference map {diff_name} with shape {diff_shape}...",
                        flush=True,
                    )
                _create_zarr_array(
                    root,
                    diff_name,
                    shape=diff_shape,
                    chunks=diff_chunks,
                    dtype=diff_dtype,
                )
                if verbose:
                    print(f"    ✓ Created difference map {diff_name}", flush=True)

            if verbose:
                print(
                    f"    ✓ Created Laplacian pyramid arrays (base level: {store_base_level}, {num_pyramid_levels - 1} difference maps)",
                    flush=True,
                )
                print("\n" + "=" * 70, flush=True)
                print(
                    "  [STEP 1/1] LOADING + BUILDING LAPLACIAN PYRAMID + WRITING DIFFERENCE MAPS",
                    flush=True,
                )
                print("=" * 70 + "\n", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()

            # Setup multiprocessing
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
                    print(
                        f"    Using multiprocessing with {workers} workers",
                        flush=True,
                    )
                    print(
                        f"    Processing {len(file_list)} images: load → build Laplacian pyramid → write difference maps",
                        flush=True,
                    )
                else:
                    print("    Using sequential processing (1 worker)", flush=True)
                    print(f"    Processing {len(file_list)} images...", flush=True)

            # Prepare tasks for Laplacian pyramid worker
            if use_multiprocessing and len(file_list) > 10:
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
                            interpolation_mode,
                            store_base_level,
                            normalize,
                            global_mean,
                            global_std,
                        ),
                    )

                if verbose:
                    print(
                        f"\n    Starting multiprocessing pool with {workers} workers...",
                        flush=True,
                    )

                # Use spawn method on Linux with Python < 3.11
                if sys.platform.startswith("linux") and sys.version_info < (3, 11):
                    ctx = multiprocessing.get_context("spawn")
                    pool = ctx.Pool(processes=workers)
                else:
                    pool = multiprocessing.Pool(processes=workers)

                try:
                    if tqdm is not None:
                        write_results = list(
                            tqdm(
                                pool.imap_unordered(
                                    _load_and_write_laplacian_pyramid,
                                    tasks,
                                ),
                                total=len(tasks),
                                desc=f"  Processing {basename}",
                                unit="image",
                            ),
                        )
                    else:
                        write_results = list(
                            pool.imap_unordered(
                                _load_and_write_laplacian_pyramid, tasks
                            )
                        )

                    # Check for failures
                    failures = [r for r in write_results if not r[1]]
                    if failures:
                        if verbose:
                            print(
                                f"    Warning: {len(failures)} images failed to process",
                                flush=True,
                            )
                finally:
                    pool.close()
                    pool.join()

            else:
                # Sequential processing
                write_results = []
                for z_idx, (_, filepath) in enumerate(file_list):
                    task = (
                        z_idx,
                        filepath,
                        str(output_path),
                        pyramid_level_shapes,
                        pyramid_scale_factors,
                        dtype,
                        has_channels,
                        final_axis_order,
                        C,
                        Y,
                        X,
                        interpolation_mode,
                        store_base_level,
                        normalize,
                        global_mean,
                        global_std,
                    )
                    result = _load_and_write_laplacian_pyramid(task)
                    write_results.append(result)
                    if verbose and (z_idx + 1) % max(1, len(file_list) // 20) == 0:
                        print(
                            f"    Processed {z_idx + 1}/{len(file_list)} images...",
                            flush=True,
                        )

                failures = [r for r in write_results if not r[1]]
                if failures:
                    if verbose:
                        print(
                            f"    Warning: {len(failures)} images failed to process",
                            flush=True,
                        )

            if verbose:
                print(
                    f"\n  ✓ Completed Laplacian pyramid OME-Zarr: {basename}",
                    flush=True,
                )
                print(f"  Output: {output_path}", flush=True)
                print(f"  Total pyramid levels: {num_pyramid_levels}", flush=True)
                print(f"  Base level stored: {store_base_level}", flush=True)
                print(f"{'=' * 70}", flush=True)
        else:
            print(
                f"  Dry run: Would create Laplacian pyramid OME-Zarr at {output_path}"
            )
            print(f"  Base shape: {base_shape}, dtype: {dtype}")
            print(f"  Pyramid levels: {num_pyramid_levels}")

        # Store results
        results[basename] = {
            "zarr_path": str(output_path),
            "shape": base_shape,  # Full resolution shape
            "dtype": dtype,
            "file_count": len(file_list),
            "files": [str(f) for _, f in file_list],
            "counter_range": (counter_min, counter_max),
            "axis_order": final_axis_order,
            "pyramid_levels": num_pyramid_levels,
        }

    if verbose:
        print(f"\n{'=' * 70}")
        print(
            f"✓ Successfully processed {len(results)} stack(s) as Laplacian pyramid OME-Zarr"
        )
        for stack_name, metadata in results.items():
            print(f"  - {stack_name}: {metadata['zarr_path']}")
            print(
                f"    Shape: {metadata['shape']}, Levels: {metadata['pyramid_levels']}"
            )
    return results


def _reconstruct_slice_worker(args: tuple) -> tuple[int, np.ndarray]:
    """
    Worker function for parallel reconstruction of a single slice (returns in-memory).

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - z_idx: int - Z-index to reconstruct
        - zarr_group_path: str - Path to zarr group
        - has_channels: bool - Whether image has channels
        - axis_order: str | None - Axis order for multi-channel
        - interpolation_mode: str - Interpolation mode
        - target_level: int | None - Target pyramid level

    Returns
    -------
    tuple[int, np.ndarray]
        (z_idx, reconstructed_slice) tuple
    """
    (
        z_idx,
        zarr_group_path,
        has_channels,
        axis_order,
        interpolation_mode,
        target_level,
    ) = args

    # Open zarr group in read mode (thread-safe for reading)
    zarr_group = zarr.open_group(str(zarr_group_path), mode="r")

    # Reconstruct this slice
    reconstructed = _reconstruct_slice_from_laplacian(
        zarr_group,
        z_idx,
        has_channels,
        axis_order,
        interpolation_mode,
        target_level,
    )

    return (z_idx, reconstructed)


def _reconstruct_and_write_slice_worker(args: tuple) -> tuple[int, bool]:
    """
    Worker function for parallel reconstruction and direct writing to zarr.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - z_idx: int - Z-index to reconstruct
        - input_zarr_group_path: str - Path to input zarr group (Laplacian pyramid)
        - output_zarr_array_path: str - Path to output zarr array
        - has_channels: bool - Whether image has channels
        - axis_order: str | None - Axis order for multi-channel
        - interpolation_mode: str - Interpolation mode
        - target_level: int | None - Target pyramid level

    Returns
    -------
    tuple[int, bool]
        (z_idx, success) tuple
    """
    (
        z_idx,
        input_zarr_group_path,
        output_zarr_array_path,
        has_channels,
        axis_order,
        interpolation_mode,
        target_level,
    ) = args

    try:
        # Open input zarr group in read mode
        input_zarr_group = zarr.open_group(str(input_zarr_group_path), mode="r")

        # Reconstruct this slice
        reconstructed = _reconstruct_slice_from_laplacian(
            input_zarr_group,
            z_idx,
            has_channels,
            axis_order,
            interpolation_mode,
            target_level,
        )

        # Open output zarr array in read-write mode (supports concurrent writes)
        output_array = zarr.open_array(str(output_zarr_array_path), mode="r+")

        # Write directly to zarr
        if has_channels:
            if axis_order == "ZCYX":
                output_array[z_idx, :, :, :] = reconstructed
            else:  # CZYX
                output_array[:, z_idx, :, :] = reconstructed
        else:
            output_array[z_idx, :, :] = reconstructed

        return (z_idx, True)
    except Exception:
        import traceback

        traceback.print_exc()
        return (z_idx, False)


def reconstruct_from_laplacian_pyramid(
    zarr_group_path: str | Path,
    z_idx: int | None = None,
    interpolation_mode: str = "bilinear",
    target_level: int | None = None,
    num_workers: int | None = None,
    output_zarr_path: str | Path | None = None,
    zarr_chunks: tuple[int, ...] | None = None,
    verbose: bool = False,
) -> np.ndarray | str:
    """
    Reconstruct image from Laplacian pyramid to a specific resolution level.

    Reconstructs the image by starting from the base level (lowest resolution)
    and progressively adding difference maps while upsampling.

    Parameters
    ----------
    zarr_group_path : str | Path
        Path to OME-Zarr group containing Laplacian pyramid
    z_idx : int | None
        Z-index to reconstruct. If None, reconstructs all slices (returns full stack)
    interpolation_mode : str
        Interpolation mode for upsampling: "bilinear" or "bicubic"
    target_level : int | None
        Target pyramid level to reconstruct to. Level 0 = full resolution (highest).
        If None, reconstructs to full resolution (level 0).
        Examples:
        - target_level=0: Full resolution (adds all diff maps)
        - target_level=1: Level 1 resolution (adds diff_1, diff_2, ... but not diff_0)
        - target_level=2: Level 2 resolution (adds diff_2, diff_3, ... but not diff_0, diff_1)
    num_workers : int | None
        Number of parallel workers for reconstruction. If None, uses all available CPU cores.
        If 1, uses sequential processing. Only used when z_idx is None (reconstructing all slices).
    output_zarr_path : str | Path | None
        Path to output zarr array for writing reconstructed stack. If provided and z_idx is None,
        writes directly to zarr file in parallel instead of returning in-memory array.
        If None, returns numpy array in memory.
    zarr_chunks : tuple[int, ...] | None
        Chunk size for output zarr array. If None, uses default chunking.
        Only used when output_zarr_path is provided.
    verbose : bool
        Whether to print progress messages. Default: False.

    Returns
    -------
    np.ndarray | str
        - If output_zarr_path is None: Returns reconstructed image(s) as np.ndarray
        - If output_zarr_path is provided and z_idx is None: Returns output_zarr_path as str
        - If output_zarr_path is provided and z_idx is int: Returns reconstructed slice as np.ndarray
        Shape depends on z_idx and target_level:
        - If z_idx is None: (Z, C, Y, X) or (Z, Y, X) - full stack
        - If z_idx is int: (C, Y, X) or (Y, X) - single slice
        Resolution depends on target_level parameter.
    """
    if not HAS_TORCH:
        msg = "PyTorch is required for Laplacian pyramid reconstruction. Install with: pip install torch"
        raise ImportError(msg)

    zarr_group = zarr.open_group(str(zarr_group_path), mode="r")

    # Find base level (stored at highest level number to match standard convention)
    # Base level is the lowest resolution Gaussian level
    # Find the highest numbered level (excluding diff maps)
    numeric_levels = [int(k) for k in _get_zarr_group_keys(zarr_group) if k.isdigit()]

    if not numeric_levels:
        msg = "Base level not found in Laplacian pyramid"
        raise ValueError(msg)

    base_level_num = max(numeric_levels)
    base_array = zarr_group[str(base_level_num)]
    base_shape = base_array.shape

    # Determine if multi-channel
    if len(base_shape) == 3:  # (Z, Y, X)
        has_channels = False
        Z, Y_base, X_base = base_shape
    elif len(base_shape) == 4:  # (Z, C, Y, X) or (C, Z, Y, X)
        has_channels = True
        # Try to determine axis order from shape
        # Assume ZCYX for now (can be improved with metadata)
        if base_shape[0] > base_shape[1]:  # Likely (Z, C, Y, X)
            Z, C, Y_base, X_base = base_shape
            axis_order = "ZCYX"
        else:  # Likely (C, Z, Y, X)
            C, Z, Y_base, X_base = base_shape
            axis_order = "CZYX"
    else:
        msg = f"Unexpected base array shape: {base_shape}"
        raise ValueError(msg)

    # Find all difference map levels
    diff_levels = []
    for key in sorted(_get_zarr_group_keys(zarr_group)):
        if key.startswith("diff_"):
            level_idx = int(key.split("_")[1])
            diff_levels.append((level_idx, zarr_group[key]))

    if not diff_levels:
        msg = "No difference maps found in Laplacian pyramid"
        raise ValueError(msg)

    # Sort by level index in reverse order (highest index = lowest resolution first)
    # This ensures we process from base level up: diff_1 then diff_0
    diff_levels.sort(key=lambda x: x[0], reverse=True)

    # Determine target shape from highest difference map
    highest_diff = diff_levels[-1][1]
    if has_channels:
        if axis_order == "ZCYX":
            _, _, Y_target, X_target = highest_diff.shape
        else:  # CZYX
            _, _, Y_target, X_target = highest_diff.shape
    else:
        _, Y_target, X_target = highest_diff.shape

    if z_idx is None:
        # Reconstruct all slices
        # Setup multiprocessing for parallel reconstruction
        if num_workers is None:
            num_cores = multiprocessing.cpu_count()
            use_multiprocessing = num_cores > 1 and Z > 1
            workers = num_cores
        elif num_workers > 1:
            use_multiprocessing = True
            workers = num_workers
        else:
            use_multiprocessing = False
            workers = 1

        # If output_zarr_path is provided, write directly to zarr in parallel
        if output_zarr_path is not None:
            # Determine output shape
            if has_channels:
                if axis_order == "ZCYX":
                    output_shape = (Z, C, Y_target, X_target)
                else:  # CZYX
                    output_shape = (C, Z, Y_target, X_target)
            else:
                output_shape = (Z, Y_target, X_target)

            # Determine chunks
            if zarr_chunks is None:
                # Default chunking: chunk along Z dimension
                if has_channels:
                    if axis_order == "ZCYX":
                        chunks = (1, C, Y_target, X_target)
                    else:  # CZYX
                        chunks = (C, 1, Y_target, X_target)
                else:
                    chunks = (1, Y_target, X_target)
            else:
                chunks = zarr_chunks

            # Use ProcessSynchronizer for concurrent writes when using multiprocessing
            is_linux_python_old = sys.platform.startswith(
                "linux"
            ) and sys.version_info < (3, 11)
            use_synchronizer = (
                use_multiprocessing
                and ProcessSynchronizer is not None
                and not is_linux_python_old
            )

            if use_synchronizer:
                sync_path = str(
                    Path(output_zarr_path).parent / ".zarr_sync_reconstruct"
                )
                synchronizer = ProcessSynchronizer(sync_path)
                if verbose:
                    print(
                        "  Creating output zarr array with ProcessSynchronizer for parallel writes...",
                        flush=True,
                    )
                output_array = zarr.open_array(
                    str(output_zarr_path),
                    mode="w",
                    shape=output_shape,
                    dtype=base_array.dtype,
                    chunks=chunks,
                    synchronizer=synchronizer,
                )
            else:
                if verbose and use_multiprocessing:
                    if ProcessSynchronizer is None:
                        print(
                            "  Creating output zarr array (ProcessSynchronizer not available)...",
                            flush=True,
                        )
                    elif is_linux_python_old:
                        print(
                            "  Creating output zarr array (ProcessSynchronizer disabled on Linux/Python < 3.11)...",
                            flush=True,
                        )
                    else:
                        print("  Creating output zarr array...", flush=True)
                output_array = zarr.open_array(
                    str(output_zarr_path),
                    mode="w",
                    shape=output_shape,
                    dtype=base_array.dtype,
                    chunks=chunks,
                )

            if verbose:
                print(
                    f"  Output shape: {output_shape}, dtype: {base_array.dtype}",
                    flush=True,
                )
                print(f"  Chunks: {chunks}", flush=True)

            if use_multiprocessing:
                # Prepare tasks for parallel reconstruction and writing
                tasks = []
                for z in range(Z):
                    tasks.append(
                        (
                            z,
                            str(zarr_group_path),
                            str(output_zarr_path),
                            has_channels,
                            axis_order if has_channels else None,
                            interpolation_mode,
                            target_level,
                        )
                    )

                if verbose:
                    print(
                        f"  Reconstructing and writing {Z} slices using {workers} parallel workers...",
                        flush=True,
                    )

                # Use spawn method on Linux with Python < 3.11
                if sys.platform.startswith("linux") and sys.version_info < (3, 11):
                    ctx = multiprocessing.get_context("spawn")
                    pool = ctx.Pool(processes=workers)
                else:
                    pool = multiprocessing.Pool(processes=workers)

                try:
                    # Reconstruct and write slices in parallel
                    write_results = pool.map(_reconstruct_and_write_slice_worker, tasks)

                    # Check for failures
                    failures = [r for r in write_results if not r[1]]
                    if failures:
                        if verbose:
                            print(
                                f"  Warning: {len(failures)} slices failed to reconstruct/write",
                                flush=True,
                            )
                finally:
                    pool.close()
                    pool.join()
            else:
                # Sequential processing with direct write
                if verbose:
                    print(
                        f"  Reconstructing and writing {Z} slices sequentially...",
                        flush=True,
                    )
                for z in range(Z):
                    slice_recon = _reconstruct_slice_from_laplacian(
                        zarr_group,
                        z,
                        has_channels,
                        axis_order if has_channels else None,
                        interpolation_mode,
                        target_level,
                    )
                    if has_channels:
                        if axis_order == "ZCYX":
                            output_array[z, :, :, :] = slice_recon
                        else:  # CZYX
                            output_array[:, z, :, :] = slice_recon
                    else:
                        output_array[z, :, :] = slice_recon
                    if verbose and (z + 1) % max(1, Z // 20) == 0:
                        print(f"    Processed {z + 1}/{Z} slices...", flush=True)

            if verbose:
                print(f"  ✓ Reconstruction complete: {output_zarr_path}", flush=True)
            return str(output_zarr_path)
        else:
            # In-memory reconstruction (original behavior)
            if has_channels:
                if axis_order == "ZCYX":
                    reconstructed = np.zeros(
                        (Z, C, Y_target, X_target), dtype=base_array.dtype
                    )
                else:  # CZYX
                    reconstructed = np.zeros(
                        (C, Z, Y_target, X_target), dtype=base_array.dtype
                    )
            else:
                reconstructed = np.zeros(
                    (Z, Y_target, X_target), dtype=base_array.dtype
                )

            if use_multiprocessing:
                # Prepare tasks for parallel reconstruction
                tasks = []
                for z in range(Z):
                    tasks.append(
                        (
                            z,
                            str(zarr_group_path),
                            has_channels,
                            axis_order if has_channels else None,
                            interpolation_mode,
                            target_level,
                        )
                    )

                # Use spawn method on Linux with Python < 3.11
                if sys.platform.startswith("linux") and sys.version_info < (3, 11):
                    ctx = multiprocessing.get_context("spawn")
                    pool = ctx.Pool(processes=workers)
                else:
                    pool = multiprocessing.Pool(processes=workers)

                try:
                    # Reconstruct slices in parallel
                    results = pool.map(_reconstruct_slice_worker, tasks)

                    # Assemble results in correct order
                    for z, slice_recon in results:
                        if has_channels:
                            if axis_order == "ZCYX":
                                reconstructed[z] = slice_recon
                            else:  # CZYX
                                reconstructed[:, z] = slice_recon
                        else:
                            reconstructed[z] = slice_recon
                finally:
                    pool.close()
                    pool.join()
            else:
                # Sequential processing
                for z in range(Z):
                    slice_recon = _reconstruct_slice_from_laplacian(
                        zarr_group,
                        z,
                        has_channels,
                        axis_order if has_channels else None,
                        interpolation_mode,
                        target_level,
                    )
                    if has_channels:
                        if axis_order == "ZCYX":
                            reconstructed[z] = slice_recon
                        else:  # CZYX
                            reconstructed[:, z] = slice_recon
                    else:
                        reconstructed[z] = slice_recon

            return reconstructed
    else:
        # Reconstruct single slice
        return _reconstruct_slice_from_laplacian(
            zarr_group,
            z_idx,
            has_channels,
            axis_order if has_channels else None,
            interpolation_mode,
            target_level,
        )


def _reconstruct_slice_from_laplacian(
    zarr_group: zarr.Group,
    z_idx: int,
    has_channels: bool,
    axis_order: str | None,
    interpolation_mode: str,
    target_level: int | None = None,
) -> np.ndarray:
    """
    Helper function to reconstruct a single slice from Laplacian pyramid.

    Parameters
    ----------
    target_level : int | None
        Target pyramid level to reconstruct to. Level 0 = full resolution.
        If None, reconstructs to full resolution (level 0).
        Only difference maps with level_idx >= target_level will be added.
    """
    # Find base level (stored at highest level number to match standard convention)
    numeric_levels = [int(k) for k in _get_zarr_group_keys(zarr_group) if k.isdigit()]

    if not numeric_levels:
        msg = "Base level not found in Laplacian pyramid"
        raise ValueError(msg)

    base_level_num = max(numeric_levels)
    base_array = zarr_group[str(base_level_num)]
    if has_channels:
        if axis_order == "ZCYX":
            base_slice = base_array[z_idx, :, :, :]  # (C, Y, X)
        else:  # CZYX
            base_slice = base_array[:, z_idx, :, :]  # (C, Y, X)
    else:
        base_slice = base_array[z_idx, :, :]  # (Y, X)

    # Start reconstruction from base level
    reconstructed = base_slice.copy().astype(np.float32)

    # Find all difference maps for this slice
    diff_levels = []
    for key in sorted(_get_zarr_group_keys(zarr_group)):
        if key.startswith("diff_"):
            level_idx = int(key.split("_")[1])
            diff_array = zarr_group[key]
            if has_channels:
                if axis_order == "ZCYX":
                    diff_slice = diff_array[z_idx, :, :, :]  # (C, Y, X)
                else:  # CZYX
                    diff_slice = diff_array[:, z_idx, :, :]  # (C, Y, X)
            else:
                diff_slice = diff_array[z_idx, :, :]  # (Y, X)
            diff_levels.append((level_idx, diff_slice))

    # Sort by level index in reverse order (highest index = lowest resolution first)
    # This ensures we process from base level up: diff_1 (32x32) then diff_0 (64x64)
    diff_levels.sort(key=lambda x: x[0], reverse=True)

    # Filter difference maps based on target_level
    # target_level=0 means full resolution (use all diff maps)
    # target_level=N means only use diff maps with level_idx >= N
    if target_level is not None:
        diff_levels = [
            (level_idx, diff_map)
            for level_idx, diff_map in diff_levels
            if level_idx >= target_level
        ]

    # Progressively upsample and add difference maps (from lowest to highest resolution)
    for _level_idx, diff_map in diff_levels:
        # Upsample current reconstruction to match difference map size
        target_size = diff_map.shape[-2:]  # (Y, X)
        upsampled = _upsample_with_torch(
            reconstructed.astype(np.float32),
            target_size,
            mode=interpolation_mode,
        )

        # Add difference map
        reconstructed = upsampled + diff_map.astype(np.float32)

    # Convert back to original dtype
    return reconstructed.astype(base_slice.dtype)
