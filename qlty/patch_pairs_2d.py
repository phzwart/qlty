"""
Extract pairs of patches from 2D image tensors with controlled displacement.

This module provides functionality to extract pairs of patches from 2D tensors
where the displacement between patch centers follows specified constraints.
"""

import multiprocessing
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import zarr



def extract_patch_pairs(
    tensor: torch.Tensor,
    window: Tuple[int, int],
    num_patches: int,
    delta_range: Tuple[float, float],
    random_seed: Optional[int] = None,
    rotation_choices: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract pairs of patches from 2D image tensors with controlled displacement.

    For each image in the input tensor, this function extracts P pairs of patches.
    Each pair consists of two patches: one at location (x_i, y_i) and another at
    (x_i + dx_i, y_i + dy_i), where the Euclidean distance between the locations
    is constrained to be within the specified delta_range.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (N, C, Y, X) where:
        - N: Number of images
        - C: Number of channels
        - Y: Height of images
        - X: Width of images
    window : Tuple[int, int]
        Window shape (U, V) where:
        - U: Height of patches
        - V: Width of patches
    num_patches : int
        Number of patch pairs P to extract per image
    delta_range : Tuple[float, float]
        Range (low, high) for the Euclidean distance of displacement vectors.
        The constraint is: low <= sqrt(dx_i² + dy_i²) <= high
        Additionally, low and high must satisfy: window//4 <= low <= high <= 3*window//4
        where window is the maximum of U and V.
    random_seed : Optional[int], optional
        Random seed for reproducibility. If None, uses current random state.
        Default is None.
    rotation_choices : Optional[Sequence[int]], optional
        Allowed quarter-turn rotations (0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°) to apply
        to the second patch in each pair. If provided, a rotation from this set is
        sampled uniformly per pair and tracked in the returned `rotations` tensor.
        When None (default), no rotations are applied.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - patches1: Tensor of shape (N*P, C, U, V) containing patches at (x_i, y_i)
        - patches2: Tensor of shape (N*P, C, U, V) containing patches at (x_i + dx_i, y_i + dy_i)
        - deltas: Tensor of shape (N*P, 2) containing (dx_i, dy_i) displacement vectors
        - rotations: Tensor of shape (N*P,) containing quarter-turn rotations applied to patches2

    Raises
    ------
    ValueError
        If delta_range constraints are violated or image dimensions are too small
        for the specified window and delta range.

    Examples
    --------
    >>> tensor = torch.randn(5, 3, 128, 128)  # 5 images, 3 channels, 128x128
    >>> window = (32, 32)  # 32x32 patches
    >>> num_patches = 10  # 10 patch pairs per image
    >>> delta_range = (8.0, 16.0)  # Euclidean distance between 8 and 16 pixels
    >>> patches1, patches2, deltas, rotations = extract_patch_pairs(
    ...     tensor, window, num_patches, delta_range
    ... )
    >>> print(patches1.shape)   # (50, 3, 32, 32)
    >>> print(patches2.shape)   # (50, 3, 32, 32)
    >>> print(deltas.shape)     # (50, 2)
    >>> print(rotations.shape)  # (50,)
    """
    # Validate input tensor shape
    if len(tensor.shape) != 4:
        raise ValueError(
            f"Input tensor must be 4D (N, C, Y, X), got shape {tensor.shape}"
        )

    N, C, Y, X = tensor.shape
    U, V = window

    # Validate delta_range constraints
    max_window = max(U, V)
    window_quarter = max_window // 4
    window_three_quarters = 3 * max_window // 4

    low, high = delta_range
    if low < window_quarter or high > window_three_quarters:
        raise ValueError(
            f"delta_range must satisfy: {window_quarter} <= low <= high <= {window_three_quarters}, "
            f"got ({low}, {high})"
        )
    if low > high:
        raise ValueError(f"delta_range low ({low}) must be <= high ({high})")

    # Check if image is large enough for window and delta range
    min_y = U + int(high)
    min_x = V + int(high)
    if Y < min_y or X < min_x:
        raise ValueError(
            f"Image dimensions ({Y}, {X}) are too small for window ({U}, {V}) "
            f"and delta_range ({low}, {high}). Minimum required: ({min_y}, {min_x})"
        )

    # Set random seed if provided
    if random_seed is not None:
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(random_seed)
    else:
        generator = None

    # Pre-allocate output tensors
    total_patches = N * num_patches
    patches1 = torch.empty(
        (total_patches, C, U, V), dtype=tensor.dtype, device=tensor.device
    )
    patches2 = torch.empty(
        (total_patches, C, U, V), dtype=tensor.dtype, device=tensor.device
    )
    deltas_tensor = torch.empty(
        (total_patches, 2), dtype=torch.float32, device=tensor.device
    )
    rotations_tensor = torch.zeros(
        total_patches, dtype=torch.int64, device=tensor.device
    )

    if rotation_choices is None:
        rotation_choices = (0,)
    else:
        rotation_choices = tuple(int(choice) % 4 for choice in rotation_choices)
        if len(rotation_choices) == 0:
            rotation_choices = (0,)
    rotation_choices_tensor = torch.tensor(
        rotation_choices, dtype=torch.int64, device=tensor.device
    )
    allow_rotations = any(choice != 0 for choice in rotation_choices)

    patch_idx = 0

    # Process each image
    for n in range(N):
        image = tensor[n]  # Shape: (C, Y, X)

        # Extract P patch pairs for this image
        for _ in range(num_patches):
            # Sample displacement vector (dx, dy) with Euclidean distance constraint
            dx, dy = _sample_displacement_vector(
                low, high, generator, device=tensor.device
            )

            # Sample first patch location (x, y) ensuring both patches fit
            # Valid x range: [0, X - V - max(|dx|, 0)]
            # Valid y range: [0, Y - U - max(|dy|, 0)]
            # But we need to ensure both patches fit, so:
            # x in [max(0, -dx), min(X - V, X - V - dx)]
            # y in [max(0, -dy), min(Y - U, Y - U - dy)]

            x_min = max(0, -dx)
            x_max = min(X - V, X - V - dx)
            y_min = max(0, -dy)
            y_max = min(Y - U, Y - U - dy)

            if x_min >= x_max or y_min >= y_max:
                # If displacement is too large, try again with a smaller one
                # This shouldn't happen often if delta_range is reasonable
                attempts = 0
                while (x_min >= x_max or y_min >= y_max) and attempts < 10:
                    dx, dy = _sample_displacement_vector(
                        low, high, generator, device=tensor.device
                    )
                    x_min = max(0, -dx)
                    x_max = min(X - V, X - V - dx)
                    y_min = max(0, -dy)
                    y_max = min(Y - U, Y - U - dy)
                    attempts += 1

                if x_min >= x_max or y_min >= y_max:
                    raise ValueError(
                        f"Could not find valid patch locations for displacement ({dx}, {dy}) "
                        f"in image of size ({Y}, {X}) with window ({U}, {V})"
                    )

            # Sample random location for first patch (keep on GPU if possible)
            if generator is not None:
                x = torch.randint(
                    x_min, x_max, (1,), generator=generator, device=tensor.device
                )[0]
                y = torch.randint(
                    y_min, y_max, (1,), generator=generator, device=tensor.device
                )[0]
            else:
                x = torch.randint(x_min, x_max, (1,), device=tensor.device)[0]
                y = torch.randint(y_min, y_max, (1,), device=tensor.device)[0]

            # Convert to Python int for slicing (necessary for indexing)
            x_int = int(x)
            y_int = int(y)

            # Extract first patch at (x, y)
            patch1 = image[:, y_int : y_int + U, x_int : x_int + V]  # Shape: (C, U, V)

            # Extract second patch at (x + dx, y + dy)
            patch2 = image[
                :, y_int + dy : y_int + dy + U, x_int + dx : x_int + dx + V
            ]  # Shape: (C, U, V)

            if allow_rotations:
                rotation_idx_tensor = torch.randint(
                    0,
                    rotation_choices_tensor.numel(),
                    (1,),
                    generator=generator,
                    device=tensor.device,
                )[0]
                rotation_idx = int(rotation_idx_tensor)
                rotation = int(rotation_choices_tensor[rotation_idx])
            else:
                rotation = 0

            if rotation != 0:
                patch2 = torch.rot90(patch2, k=rotation, dims=(-2, -1))

            # Store patches and delta directly in pre-allocated tensors
            patches1[patch_idx] = patch1
            patches2[patch_idx] = patch2
            deltas_tensor[patch_idx, 0] = float(dx)
            deltas_tensor[patch_idx, 1] = float(dy)
            rotations_tensor[patch_idx] = rotation

            patch_idx += 1

    return patches1, patches2, deltas_tensor, rotations_tensor


def _sample_displacement_vector(
    low: float,
    high: float,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, int]:
    """
    Sample a displacement vector (dx, dy) such that low <= sqrt(dx² + dy²) <= high.

    Uses rejection sampling to ensure the Euclidean distance constraint is satisfied.

    Parameters
    ----------
    low : float
        Minimum Euclidean distance
    high : float
        Maximum Euclidean distance
    generator : Optional[torch.Generator]
        Random number generator for reproducibility

    Returns
    -------
    Tuple[int, int]
        Displacement vector (dx, dy) as integers
    """
    max_attempts = 1000
    for _ in range(max_attempts):
        # Sample dx and dy in a range that could potentially satisfy the constraint
        # We sample from a larger range to ensure we can find valid vectors
        max_delta = int(high) + 1

        if device is None:
            device = torch.device("cpu")

        if generator is not None:
            dx_tensor = torch.randint(
                -max_delta, max_delta + 1, (1,), generator=generator, device=device
            )
            dy_tensor = torch.randint(
                -max_delta, max_delta + 1, (1,), generator=generator, device=device
            )
        else:
            dx_tensor = torch.randint(-max_delta, max_delta + 1, (1,), device=device)
            dy_tensor = torch.randint(-max_delta, max_delta + 1, (1,), device=device)

        dx = int(dx_tensor[0])
        dy = int(dy_tensor[0])

        # Check Euclidean distance constraint
        distance = (dx**2 + dy**2) ** 0.5
        if low <= distance <= high:
            return dx, dy

    # If we couldn't find a valid vector after many attempts, use a fallback
    # Sample angle uniformly and distance uniformly in [low, high]
    if generator is not None:
        angle_tensor = (
            torch.rand(1, generator=generator, device=device) * 2 * 3.141592653589793
        )
        distance_tensor = low + (high - low) * torch.rand(
            1, generator=generator, device=device
        )
    else:
        angle_tensor = torch.rand(1, device=device) * 2 * 3.141592653589793
        distance_tensor = low + (high - low) * torch.rand(1, device=device)

    distance = float(distance_tensor[0])

    # Compute cos and sin on GPU if device is GPU
    cos_val = torch.cos(angle_tensor)[0]
    sin_val = torch.sin(angle_tensor)[0]
    dx = int(round(distance * float(cos_val)))
    dy = int(round(distance * float(sin_val)))

    # Ensure distance is still in range (may have been affected by rounding)
    actual_distance = (dx**2 + dy**2) ** 0.5
    if actual_distance < low:
        # Scale up to meet minimum
        scale = low / actual_distance
        dx = int(round(dx * scale))
        dy = int(round(dy * scale))
    elif actual_distance > high:
        # Scale down to meet maximum
        scale = high / actual_distance
        dx = int(round(dx * scale))
        dy = int(round(dy * scale))

    return dx, dy


def extract_overlapping_pixels(
    patches1: torch.Tensor,
    patches2: torch.Tensor,
    deltas: torch.Tensor,
    rotations: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract overlapping pixels from patch pairs based on displacement vectors.

    For each patch pair, this function finds pixels that have valid correspondences
    between the two patches (i.e., pixels that represent the same spatial location
    in the original image). Only overlapping pixels are returned.

    Parameters
    ----------
    patches1 : torch.Tensor
        First set of patches, shape (N*P, C, U, V) where:
        - N*P: Total number of patch pairs
        - C: Number of channels
        - U: Patch height
        - V: Patch width
    patches2 : torch.Tensor
        Second set of patches, shape (N*P, C, U, V), corresponding patches
        extracted at displaced locations
    deltas : torch.Tensor
        Displacement vectors, shape (N*P, 2) containing (dx, dy) for each pair
    rotations : Optional[torch.Tensor], optional
        Quarter-turn rotations (0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°) that were
        applied to `patches2`. When provided, each value is used to undo the rotation
        before extracting overlaps so that corresponding pixels align spatially.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - overlapping1: Overlapping pixel values from patches1, shape (K, C)
        - overlapping2: Overlapping pixel values from patches2, shape (K, C)
        where K is the total number of overlapping pixels across all patch pairs,
        and corresponding pixels are at the same index in both tensors.

    Examples
    --------
    >>> patches1 = torch.randn(10, 3, 32, 32)
    >>> patches2 = torch.randn(10, 3, 32, 32)
    >>> deltas = torch.tensor([[5, 3], [-2, 4], ...])  # shape (10, 2)
    >>> overlapping1, overlapping2 = extract_overlapping_pixels(patches1, patches2, deltas)
    >>> print(overlapping1.shape)  # (K, 3) where K depends on overlap
    >>> print(overlapping2.shape)  # (K, 3)
    >>> # overlapping1[i] and overlapping2[i] correspond to the same spatial location
    """
    # Validate input shapes
    if len(patches1.shape) != 4 or len(patches2.shape) != 4:
        raise ValueError(
            f"Both patches1 and patches2 must be 4D tensors (N*P, C, U, V), "
            f"got shapes {patches1.shape} and {patches2.shape}"
        )

    if patches1.shape != patches2.shape:
        raise ValueError(
            f"patches1 and patches2 must have the same shape, "
            f"got {patches1.shape} and {patches2.shape}"
        )

    if len(deltas.shape) != 2 or deltas.shape[1] != 2:
        raise ValueError(
            f"deltas must be 2D tensor of shape (N*P, 2), got {deltas.shape}"
        )

    num_pairs, C, U, V = patches1.shape

    if deltas.shape[0] != num_pairs:
        raise ValueError(
            f"Number of deltas ({deltas.shape[0]}) must match number of patch pairs ({num_pairs})"
        )

    if rotations is not None:
        if rotations.shape[0] != num_pairs:
            raise ValueError(
                f"Number of rotations ({rotations.shape[0]}) must match number of patch pairs ({num_pairs})"
            )
        rotations_int = rotations.int()
    else:
        rotations_int = None

    # Convert deltas to integers for indexing (keep on same device)
    deltas_int = deltas.int()

    # Collect all overlapping pixels from both patches
    overlapping_pixels1 = []
    overlapping_pixels2 = []

    for i in range(num_pairs):
        # Get delta values without moving to CPU (use indexing, then convert to int)
        dx_tensor = deltas_int[i, 0]
        dy_tensor = deltas_int[i, 1]
        # Convert to Python int only when needed for indexing
        dx = int(dx_tensor)
        dy = int(dy_tensor)

        # Get the two patches
        patch1 = patches1[i]  # Shape: (C, U, V)
        patch2 = patches2[i]  # Shape: (C, U, V)
        rotation = 0
        if rotations_int is not None:
            rotation = int(rotations_int[i] % 4)
            if rotation != 0:
                patch2 = torch.rot90(patch2, k=-rotation, dims=(-2, -1))

        # Find valid overlap region in patch1 coordinates
        # A pixel at (u1, v1) in patch1 corresponds to (u1 - dy, v1 - dx) in patch2
        # For valid correspondence, we need:
        #   0 <= u1 - dy < U  and  0 <= v1 - dx < V
        # Which means: dy <= u1 < U + dy  and  dx <= v1 < V + dx
        # Combined with u1 in [0, U) and v1 in [0, V):
        u_min = max(0, dy)
        u_max = min(U, U + dy)
        v_min = max(0, dx)
        v_max = min(V, V + dx)

        # Check if there's any overlap
        if u_min >= u_max or v_min >= v_max:
            # No overlap for this patch pair, skip it
            continue

        # Extract overlapping region from patch1
        overlap_region1 = patch1[
            :, u_min:u_max, v_min:v_max
        ]  # Shape: (C, u_max-u_min, v_max-v_min)

        # Extract corresponding region from patch2
        # In patch2 coordinates: u2 = u1 - dy, v2 = v1 - dx
        # So: u2_min = u_min - dy, u2_max = u_max - dy
        #     v2_min = v_min - dx, v2_max = v_max - dx
        u2_min = u_min - dy
        u2_max = u_max - dy
        v2_min = v_min - dx
        v2_max = v_max - dx

        overlap_region2 = patch2[
            :, u2_min:u2_max, v2_min:v2_max
        ]  # Shape: (C, u_max-u_min, v_max-v_min)

        # Reshape to (C, K') where K' is the number of overlapping pixels for this pair
        K_prime = (u_max - u_min) * (v_max - v_min)
        overlap_flat1 = overlap_region1.reshape(C, K_prime).t()  # Shape: (K', C)
        overlap_flat2 = overlap_region2.reshape(C, K_prime).t()  # Shape: (K', C)

        overlapping_pixels1.append(overlap_flat1)
        overlapping_pixels2.append(overlap_flat2)

    # Concatenate all overlapping pixels
    if len(overlapping_pixels1) == 0:
        # No overlapping pixels found, return empty tensors with correct shape
        empty_tensor = torch.empty((0, C), dtype=patches1.dtype, device=patches1.device)
        return empty_tensor, empty_tensor

    # Stack all overlapping pixels
    result1 = torch.cat(overlapping_pixels1, dim=0)  # Shape: (K, C) where K is total
    result2 = torch.cat(overlapping_pixels2, dim=0)  # Shape: (K, C) where K is total
    return result1, result2


def _process_image_for_metadata(
    args: Tuple[int, torch.Tensor, Tuple[int, int], int, Tuple[float, float], Optional[int], Optional[Sequence[int]]]
) -> Dict[str, torch.Tensor]:
    """
    Process a single image to generate patch metadata (locations and statistics).

    This is a helper function for multiprocessing in extract_patch_pairs_metadata.

    Parameters
    ----------
    args : Tuple
        Tuple containing:
        - image_idx: int, index of the image in the batch
        - image: torch.Tensor, shape (C, Y, X)
        - window: Tuple[int, int], patch window size (U, V)
        - num_patches: int, number of patches to extract
        - delta_range: Tuple[float, float], displacement range
        - random_seed: Optional[int], random seed (offset by image_idx)
        - rotation_choices: Optional[Sequence[int]], allowed rotations

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing metadata tensors for this image
    """
    image_idx, image, window, num_patches, delta_range, random_seed, rotation_choices = args

    C, Y, X = image.shape
    U, V = window
    low, high = delta_range

    # Create generator with offset seed for reproducibility
    if random_seed is not None:
        generator = torch.Generator(device=image.device)
        generator.manual_seed(random_seed + image_idx)
    else:
        generator = None

    # Pre-allocate metadata tensors
    image_idx_tensor = torch.full((num_patches,), image_idx, dtype=torch.int64, device=image.device)
    patch1_y = torch.empty(num_patches, dtype=torch.int64, device=image.device)
    patch1_x = torch.empty(num_patches, dtype=torch.int64, device=image.device)
    patch2_y = torch.empty(num_patches, dtype=torch.int64, device=image.device)
    patch2_x = torch.empty(num_patches, dtype=torch.int64, device=image.device)
    deltas = torch.empty((num_patches, 2), dtype=torch.float32, device=image.device)
    rotations = torch.zeros(num_patches, dtype=torch.int64, device=image.device)
    mean1 = torch.empty(num_patches, dtype=torch.float32, device=image.device)
    sigma1 = torch.empty(num_patches, dtype=torch.float32, device=image.device)
    mean2 = torch.empty(num_patches, dtype=torch.float32, device=image.device)
    sigma2 = torch.empty(num_patches, dtype=torch.float32, device=image.device)

    if rotation_choices is None:
        rotation_choices = (0,)
    else:
        rotation_choices = tuple(int(choice) % 4 for choice in rotation_choices)
        if len(rotation_choices) == 0:
            rotation_choices = (0,)
    rotation_choices_tensor = torch.tensor(
        rotation_choices, dtype=torch.int64, device=image.device
    )
    allow_rotations = any(choice != 0 for choice in rotation_choices)

    for p in range(num_patches):
        # Sample displacement vector
        dx, dy = _sample_displacement_vector(low, high, generator, device=image.device)

        # Compute valid patch location ranges
        x_min = max(0, -dx)
        x_max = min(X - V, X - V - dx)
        y_min = max(0, -dy)
        y_max = min(Y - U, Y - U - dy)

        if x_min >= x_max or y_min >= y_max:
            attempts = 0
            while (x_min >= x_max or y_min >= y_max) and attempts < 10:
                dx, dy = _sample_displacement_vector(low, high, generator, device=image.device)
                x_min = max(0, -dx)
                x_max = min(X - V, X - V - dx)
                y_min = max(0, -dy)
                y_max = min(Y - U, Y - U - dy)
                attempts += 1

            if x_min >= x_max or y_min >= y_max:
                raise ValueError(
                    f"Could not find valid patch locations for displacement ({dx}, {dy}) "
                    f"in image {image_idx} of size ({Y}, {X}) with window ({U}, {V})"
                )

        # Sample random location for first patch
        if generator is not None:
            x = torch.randint(x_min, x_max, (1,), generator=generator, device=image.device)[0]
            y = torch.randint(y_min, y_max, (1,), generator=generator, device=image.device)[0]
        else:
            x = torch.randint(x_min, x_max, (1,), device=image.device)[0]
            y = torch.randint(y_min, y_max, (1,), device=image.device)[0]

        x_int = int(x)
        y_int = int(y)

        # Extract patches to compute statistics
        patch1 = image[:, y_int : y_int + U, x_int : x_int + V]  # Shape: (C, U, V)
        patch2 = image[
            :, y_int + dy : y_int + dy + U, x_int + dx : x_int + dx + V
        ]  # Shape: (C, U, V)

        # Apply rotation if needed
        if allow_rotations:
            rotation_idx_tensor = torch.randint(
                0,
                rotation_choices_tensor.numel(),
                (1,),
                generator=generator,
                device=image.device,
            )[0]
            rotation_idx = int(rotation_idx_tensor)
            rotation = int(rotation_choices_tensor[rotation_idx])
        else:
            rotation = 0

        if rotation != 0:
            patch2 = torch.rot90(patch2, k=rotation, dims=(-2, -1))

        # Compute mean and sigma across all channels and spatial dimensions
        # Flatten to (C*U*V,) then compute stats
        patch1_flat = patch1.flatten()  # Shape: (C*U*V,)
        patch2_flat = patch2.flatten()  # Shape: (C*U*V,)

        mean1[p] = patch1_flat.mean().item()
        sigma1[p] = patch1_flat.std().item()
        mean2[p] = patch2_flat.mean().item()
        sigma2[p] = patch2_flat.std().item()

        # Store metadata
        patch1_y[p] = y_int
        patch1_x[p] = x_int
        patch2_y[p] = y_int + dy
        patch2_x[p] = x_int + dx
        deltas[p, 0] = float(dx)
        deltas[p, 1] = float(dy)
        rotations[p] = rotation

    return {
        "image_idx": image_idx_tensor,
        "patch1_y": patch1_y,
        "patch1_x": patch1_x,
        "patch2_y": patch2_y,
        "patch2_x": patch2_x,
        "dx": deltas[:, 0],
        "dy": deltas[:, 1],
        "rotation": rotations,
        "mean1": mean1,
        "sigma1": sigma1,
        "mean2": mean2,
        "sigma2": sigma2,
        "window": window,  # Store window for later extraction
    }


def extract_patch_pairs_metadata(
    tensor: torch.Tensor,
    window: Tuple[int, int],
    num_patches: int,
    delta_range: Tuple[float, float],
    random_seed: Optional[int] = None,
    rotation_choices: Optional[Sequence[int]] = None,
    num_workers: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract patch pair metadata (locations and statistics) without loading full patches.

    This function performs a "dry-run" that generates patch locations and computes
    mean/sigma statistics for each patch, enabling memory-efficient stratified sampling.
    Uses multiprocessing to parallelize across images (z-slices).

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (N, C, Y, X) where:
        - N: Number of images
        - C: Number of channels
        - Y: Height of images
        - X: Width of images
    window : Tuple[int, int]
        Window shape (U, V) where:
        - U: Height of patches
        - V: Width of patches
    num_patches : int
        Number of patch pairs P to extract per image
    delta_range : Tuple[float, float]
        Range (low, high) for the Euclidean distance of displacement vectors.
        The constraint is: low <= sqrt(dx_i² + dy_i²) <= high
        Additionally, low and high must satisfy: window//4 <= low <= high <= 3*window//4
        where window is the maximum of U and V.
    random_seed : Optional[int], optional
        Random seed for reproducibility. If None, uses current random state.
        Default is None.
    rotation_choices : Optional[Sequence[int]], optional
        Allowed quarter-turn rotations (0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°) to apply
        to the second patch in each pair. If provided, a rotation from this set is
        sampled uniformly per pair and tracked in the returned `rotations` tensor.
        When None (default), no rotations are applied.
    num_workers : Optional[int], optional
        Number of worker processes for multiprocessing. If None, uses all available CPUs.
        If 1, processes sequentially. Default is None.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing metadata tensors, all of shape (N*P,):
        - image_idx: Image index for each patch pair
        - patch1_y, patch1_x: Coordinates of first patch
        - patch2_y, patch2_x: Coordinates of second patch
        - dx, dy: Displacement vectors
        - rotation: Rotation applied to patch2
        - mean1, sigma1: Mean and std dev of patch1 (across all channels)
        - mean2, sigma2: Mean and std dev of patch2 (across all channels)

    Raises
    ------
    ValueError
        If delta_range constraints are violated or image dimensions are too small
        for the specified window and delta range.

    Examples
    --------
    >>> tensor = torch.randn(5, 3, 128, 128)  # 5 images, 3 channels, 128x128
    >>> window = (32, 32)  # 32x32 patches
    >>> num_patches = 10  # 10 patch pairs per image
    >>> delta_range = (8.0, 16.0)  # Euclidean distance between 8 and 16 pixels
    >>> metadata = extract_patch_pairs_metadata(
    ...     tensor, window, num_patches, delta_range, random_seed=42
    ... )
    >>> print(metadata["mean1"].shape)  # (50,)
    >>> print(metadata["image_idx"].shape)  # (50,)
    """
    # Validate input tensor shape
    if len(tensor.shape) != 4:
        raise ValueError(
            f"Input tensor must be 4D (N, C, Y, X), got shape {tensor.shape}"
        )

    N, C, Y, X = tensor.shape
    U, V = window

    # Validate delta_range constraints (same as extract_patch_pairs)
    max_window = max(U, V)
    window_quarter = max_window // 4
    window_three_quarters = 3 * max_window // 4

    low, high = delta_range
    if low < window_quarter or high > window_three_quarters:
        raise ValueError(
            f"delta_range must satisfy: {window_quarter} <= low <= high <= {window_three_quarters}, "
            f"got ({low}, {high})"
        )
    if low > high:
        raise ValueError(f"delta_range low ({low}) must be <= high ({high})")

    # Check if image is large enough
    min_y = U + int(high)
    min_x = V + int(high)
    if Y < min_y or X < min_x:
        raise ValueError(
            f"Image dimensions ({Y}, {X}) are too small for window ({U}, {V}) "
            f"and delta_range ({low}, {high}). Minimum required: ({min_y}, {min_x})"
        )

    # Determine number of workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    elif num_workers < 1:
        num_workers = 1

    # Prepare arguments for each image
    # Note: We need to move tensors to CPU for multiprocessing if they're on GPU
    device = tensor.device
    if device.type == "cuda":
        # For GPU tensors, we'll process sequentially or move to CPU
        # Multiprocessing with CUDA tensors is complex, so we'll process sequentially
        num_workers = 1

    # Prepare tasks
    tasks = []
    for n in range(N):
        image = tensor[n].cpu() if device.type == "cuda" else tensor[n]
        tasks.append((n, image, window, num_patches, delta_range, random_seed, rotation_choices))

    # Process images
    if num_workers > 1 and N > 1:
        # Use multiprocessing
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_process_image_for_metadata, tasks)
    else:
        # Sequential processing
        results = [_process_image_for_metadata(task) for task in tasks]

    # Concatenate results from all images
    total_patches = N * num_patches
    metadata = {
        "image_idx": torch.cat([r["image_idx"] for r in results], dim=0),
        "patch1_y": torch.cat([r["patch1_y"] for r in results], dim=0),
        "patch1_x": torch.cat([r["patch1_x"] for r in results], dim=0),
        "patch2_y": torch.cat([r["patch2_y"] for r in results], dim=0),
        "patch2_x": torch.cat([r["patch2_x"] for r in results], dim=0),
        "dx": torch.cat([r["dx"] for r in results], dim=0),
        "dy": torch.cat([r["dy"] for r in results], dim=0),
        "rotation": torch.cat([r["rotation"] for r in results], dim=0),
        "mean1": torch.cat([r["mean1"] for r in results], dim=0),
        "sigma1": torch.cat([r["sigma1"] for r in results], dim=0),
        "mean2": torch.cat([r["mean2"] for r in results], dim=0),
        "sigma2": torch.cat([r["sigma2"] for r in results], dim=0),
    }

    # Move back to original device if needed
    if device.type == "cuda":
        for key in metadata:
            if key != "window":  # window is a tuple, not a tensor
                metadata[key] = metadata[key].to(device)

    # Store window in metadata (from first result, all should have same window)
    metadata["window"] = results[0]["window"]

    return metadata


def extract_patches_from_metadata(
    tensor: torch.Tensor,
    metadata: Dict[str, Union[torch.Tensor, Tuple[int, int]]],
    selected_indices: Union[torch.Tensor, Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract patch pairs from tensor using pre-computed metadata and selected indices.

    This function extracts only the patches specified by selected_indices, enabling
    memory-efficient extraction after stratified sampling.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (N, C, Y, X) from which patches were originally sampled
    metadata : Dict[str, Union[torch.Tensor, Tuple[int, int]]]
        Metadata dictionary returned by extract_patch_pairs_metadata()
    selected_indices : Union[torch.Tensor, Sequence[int]]
        Indices of patches to extract. Can be a torch.Tensor or list/array of integers.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - patches1: Tensor of shape (len(selected_indices), C, U, V)
        - patches2: Tensor of shape (len(selected_indices), C, U, V)
        - deltas: Tensor of shape (len(selected_indices), 2)
        - rotations: Tensor of shape (len(selected_indices),)

    Examples
    --------
    >>> metadata = extract_patch_pairs_metadata(tensor, window, num_patches, delta_range)
    >>> selected = uniform_manifold_sample(means, sigmas, n_bins=20)
    >>> patches1, patches2, deltas, rotations = extract_patches_from_metadata(
    ...     tensor, metadata, selected
    ... )
    """
    # Convert selected_indices to tensor if needed
    if not isinstance(selected_indices, torch.Tensor):
        selected_indices = torch.tensor(selected_indices, dtype=torch.int64, device=tensor.device)
    else:
        selected_indices = selected_indices.to(tensor.device)

    num_selected = len(selected_indices)
    N, C, Y, X = tensor.shape

    # Get window size from metadata
    if "window" not in metadata:
        raise ValueError("metadata must contain 'window' key. Use extract_patch_pairs_metadata() to generate metadata.")
    window = metadata["window"]
    U, V = window

    # Pre-allocate output tensors
    patches1 = torch.empty((num_selected, C, U, V), dtype=tensor.dtype, device=tensor.device)
    patches2 = torch.empty((num_selected, C, U, V), dtype=tensor.dtype, device=tensor.device)
    deltas = torch.empty((num_selected, 2), dtype=torch.float32, device=tensor.device)
    rotations = torch.empty(num_selected, dtype=torch.int64, device=tensor.device)

    # Extract selected patches
    for i, idx in enumerate(selected_indices):
        idx_int = int(idx.item())
        image_idx = int(metadata["image_idx"][idx_int].item())
        patch1_y = int(metadata["patch1_y"][idx_int].item())
        patch1_x = int(metadata["patch1_x"][idx_int].item())
        patch2_y = int(metadata["patch2_y"][idx_int].item())
        patch2_x = int(metadata["patch2_x"][idx_int].item())
        dx = float(metadata["dx"][idx_int].item())
        dy = float(metadata["dy"][idx_int].item())
        rotation = int(metadata["rotation"][idx_int].item())

        image = tensor[image_idx]  # Shape: (C, Y, X)

        # Extract patches
        patch1 = image[:, patch1_y : patch1_y + U, patch1_x : patch1_x + V]  # Shape: (C, U, V)
        patch2 = image[:, patch2_y : patch2_y + U, patch2_x : patch2_x + V]  # Shape: (C, U, V)

        # Apply rotation if needed
        if rotation != 0:
            patch2 = torch.rot90(patch2, k=rotation, dims=(-2, -1))

        patches1[i] = patch1
        patches2[i] = patch2
        deltas[i, 0] = dx
        deltas[i, 1] = dy
        rotations[i] = rotation

    return patches1, patches2, deltas, rotations


def extract_patches_to_zarr(
    tensor: torch.Tensor,
    metadata: Dict[str, Union[torch.Tensor, Tuple[int, int]]],
    selected_indices: Union[torch.Tensor, Sequence[int]],
    zarr_path: Union[str, "zarr.Group"],
    zarr_chunks: Optional[Tuple[int, ...]] = None,
) -> "zarr.Group":
    """
    Extract patch pairs to a Zarr group for on-disk storage.

    Creates a Zarr group with arrays for patches1, patches2, deltas, and rotations,
    enabling memory-efficient storage and dataloader access.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (N, C, Y, X) from which patches were originally sampled
    metadata : Dict[str, Union[torch.Tensor, Tuple[int, int]]]
        Metadata dictionary returned by extract_patch_pairs_metadata()
    selected_indices : Union[torch.Tensor, Sequence[int]]
        Indices of patches to extract
    zarr_path : Union[str, zarr.Group]
        Path to zarr file/group or existing zarr group. If string, creates new group.
    zarr_chunks : Optional[Tuple[int, ...]], optional
        Chunk size for patch arrays. Default: (1, C, U, V) for patches, (1, 2) for deltas

    Returns
    -------
    zarr.Group
        Zarr group containing:
        - patches1: Array of shape (num_selected, C, U, V)
        - patches2: Array of shape (num_selected, C, U, V)
        - deltas: Array of shape (num_selected, 2)
        - rotations: Array of shape (num_selected,)
        - metadata attributes stored in group.attrs

    Examples
    --------
    >>> metadata = extract_patch_pairs_metadata(tensor, window, num_patches, delta_range)
    >>> selected = uniform_manifold_sample(means, sigmas, n_bins=20)
    >>> zarr_group = extract_patches_to_zarr(tensor, metadata, selected, "patches.zarr")
    >>> print(zarr_group["patches1"].shape)  # (num_selected, C, U, V)
    """
    if zarr is None:
        raise ImportError("zarr is required. Install with: pip install zarr")

    # Convert selected_indices to tensor if needed
    if not isinstance(selected_indices, torch.Tensor):
        selected_indices = torch.tensor(selected_indices, dtype=torch.int64, device=tensor.device)
    else:
        selected_indices = selected_indices.to(tensor.device)

    num_selected = len(selected_indices)
    N, C, Y, X = tensor.shape

    # Get window size from metadata
    if "window" not in metadata:
        raise ValueError("metadata must contain 'window' key. Use extract_patch_pairs_metadata() to generate metadata.")
    window = metadata["window"]
    U, V = window

    # Create or open zarr group
    if isinstance(zarr_path, str):
        zarr_group = zarr.open_group(zarr_path, mode="w")
    else:
        zarr_group = zarr_path

    # Determine chunk size
    if zarr_chunks is None:
        patch_chunks = (1, C, U, V)
        delta_chunks = (1, 2)
        rotation_chunks = (1,)
    else:
        patch_chunks = zarr_chunks
        delta_chunks = zarr_chunks[:2] if len(zarr_chunks) >= 2 else (1, 2)
        rotation_chunks = (zarr_chunks[0],) if len(zarr_chunks) >= 1 else (1,)

    # Convert torch dtype to numpy dtype for zarr
    # Create a dummy numpy array from tensor to get numpy dtype
    dummy_array = torch.zeros(1, dtype=tensor.dtype).cpu().numpy()
    np_dtype = dummy_array.dtype

    # Create zarr arrays
    patches1_array = zarr_group.create(
        "patches1",
        shape=(num_selected, C, U, V),
        chunks=patch_chunks,
        dtype=np_dtype,
    )
    patches2_array = zarr_group.create(
        "patches2",
        shape=(num_selected, C, U, V),
        chunks=patch_chunks,
        dtype=np_dtype,
    )
    deltas_array = zarr_group.create(
        "deltas",
        shape=(num_selected, 2),
        chunks=delta_chunks,
        dtype="float32",
    )
    rotations_array = zarr_group.create(
        "rotations",
        shape=(num_selected,),
        chunks=rotation_chunks,
        dtype="int64",
    )

    # Extract and write patches
    for i, idx in enumerate(selected_indices):
        idx_int = int(idx.item())
        image_idx = int(metadata["image_idx"][idx_int].item())
        patch1_y = int(metadata["patch1_y"][idx_int].item())
        patch1_x = int(metadata["patch1_x"][idx_int].item())
        patch2_y = int(metadata["patch2_y"][idx_int].item())
        patch2_x = int(metadata["patch2_x"][idx_int].item())
        dx = float(metadata["dx"][idx_int].item())
        dy = float(metadata["dy"][idx_int].item())
        rotation = int(metadata["rotation"][idx_int].item())

        image = tensor[image_idx]  # Shape: (C, Y, X)

        # Extract patches
        patch1 = image[:, patch1_y : patch1_y + U, patch1_x : patch1_x + V]  # Shape: (C, U, V)
        patch2 = image[:, patch2_y : patch2_y + U, patch2_x : patch2_x + V]  # Shape: (C, U, V)

        # Apply rotation if needed
        if rotation != 0:
            patch2 = torch.rot90(patch2, k=rotation, dims=(-2, -1))

        # Convert to numpy and write to zarr
        patches1_array[i] = patch1.cpu().numpy()
        patches2_array[i] = patch2.cpu().numpy()
        deltas_array[i] = [dx, dy]
        rotations_array[i] = rotation

    # Store metadata as group attributes
    zarr_group.attrs.update(
        {
            "window": window,
            "num_patches": num_selected,
            "num_channels": C,
            "patch_shape": (U, V),
        }
    )

    return zarr_group


class ZarrPatchPairDataset:
    """
    PyTorch-compatible dataset for reading patch pairs from Zarr storage.

    This class provides a clean interface for dataloader access to patch pairs
    stored in Zarr format, enabling memory-efficient training on large datasets.

    Parameters
    ----------
    zarr_path : Union[str, zarr.Group]
        Path to zarr file/group or existing zarr group
    transform : Optional[callable], optional
        Optional transform to apply to each patch pair. Should accept
        (patch1, patch2, delta, rotation) and return transformed version.
        Default is None.

    Examples
    --------
    >>> dataset = ZarrPatchPairDataset("patches.zarr")
    >>> print(len(dataset))  # Number of patch pairs
    >>> patch1, patch2, delta, rotation = dataset[0]  # Get first pair
    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for batch in loader:
    ...     batch_patch1, batch_patch2, batch_delta, batch_rotation = batch
    ...     # Process batch
    """

    def __init__(
        self,
        zarr_path: Union[str, "zarr.Group"],
        transform: Optional[callable] = None,
    ):
        if zarr is None:
            raise ImportError("zarr is required. Install with: pip install zarr")

        # Open zarr group
        if isinstance(zarr_path, str):
            self.zarr_group = zarr.open_group(zarr_path, mode="r")
        else:
            self.zarr_group = zarr_path

        # Get arrays
        self.patches1 = self.zarr_group["patches1"]
        self.patches2 = self.zarr_group["patches2"]
        self.deltas = self.zarr_group["deltas"]
        self.rotations = self.zarr_group["rotations"]

        self.transform = transform

    def __len__(self) -> int:
        """Return number of patch pairs in dataset."""
        return self.patches1.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a patch pair by index.

        Parameters
        ----------
        idx : int
            Index of patch pair to retrieve

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing:
            - patch1: Tensor of shape (C, U, V)
            - patch2: Tensor of shape (C, U, V)
            - delta: Tensor of shape (2,)
            - rotation: Tensor scalar (int64)
        """
        # Load from zarr (returns numpy arrays)
        patch1_np = self.patches1[idx]
        patch2_np = self.patches2[idx]
        delta_np = self.deltas[idx]
        rotation_np = self.rotations[idx]

        # Convert to torch tensors
        patch1 = torch.from_numpy(patch1_np)
        patch2 = torch.from_numpy(patch2_np)
        delta = torch.from_numpy(delta_np)
        rotation = torch.tensor(rotation_np, dtype=torch.int64)

        # Apply transform if provided
        if self.transform is not None:
            patch1, patch2, delta, rotation = self.transform(patch1, patch2, delta, rotation)

        return patch1, patch2, delta, rotation
