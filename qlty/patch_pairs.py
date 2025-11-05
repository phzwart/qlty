"""
Extract pairs of patches from 2D image tensors with controlled displacement.

This module provides functionality to extract pairs of patches from 2D tensors
where the displacement between patch centers follows specified constraints.
"""

from typing import Optional, Tuple

import torch


def extract_patch_pairs(
    tensor: torch.Tensor,
    window: Tuple[int, int],
    num_patches: int,
    delta_range: Tuple[float, float],
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - patches1: Tensor of shape (N*P, C, U, V) containing patches at (x_i, y_i)
        - patches2: Tensor of shape (N*P, C, U, V) containing patches at (x_i + dx_i, y_i + dy_i)
        - deltas: Tensor of shape (N*P, 2) containing (dx_i, dy_i) displacement vectors

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
    >>> patches1, patches2, deltas = extract_patch_pairs(tensor, window, num_patches, delta_range)
    >>> print(patches1.shape)  # (50, 3, 32, 32)
    >>> print(patches2.shape)  # (50, 3, 32, 32)
    >>> print(deltas.shape)    # (50, 2)
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

            # Store patches and delta directly in pre-allocated tensors
            patches1[patch_idx] = patch1
            patches2[patch_idx] = patch2
            deltas_tensor[patch_idx, 0] = float(dx)
            deltas_tensor[patch_idx, 1] = float(dy)

            patch_idx += 1

    return patches1, patches2, deltas_tensor


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
