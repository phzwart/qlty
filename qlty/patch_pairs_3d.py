"""
Extract pairs of patches from 3D image tensors with controlled displacement.

This module provides functionality to extract pairs of patches from 3D tensors
where the displacement between patch centers follows specified constraints.
"""

from typing import Optional, Tuple

import torch


def extract_patch_pairs_3d(
    tensor: torch.Tensor,
    window: Tuple[int, int, int],
    num_patches: int,
    delta_range: Tuple[float, float],
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract pairs of patches from 3D image tensors with controlled displacement.

    For each volume in the input tensor, this function extracts P pairs of patches.
    Each pair consists of two patches: one at location (x_i, y_i, z_i) and another at
    (x_i + dx_i, y_i + dy_i, z_i + dz_i), where the Euclidean distance between the
    locations is constrained to be within the specified delta_range.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (N, C, Z, Y, X) where:
        - N: Number of volumes
        - C: Number of channels
        - Z: Depth of volumes
        - Y: Height of volumes
        - X: Width of volumes
    window : Tuple[int, int, int]
        Window shape (U, V, W) where:
        - U: Depth of patches
        - V: Height of patches
        - W: Width of patches
    num_patches : int
        Number of patch pairs P to extract per volume
    delta_range : Tuple[float, float]
        Range (low, high) for the Euclidean distance of displacement vectors.
        The constraint is: low <= sqrt(dx_i² + dy_i² + dz_i²) <= high
        Additionally, low and high must satisfy: window//4 <= low <= high <= 3*window//4
        where window is the maximum of U, V, and W.
    random_seed : Optional[int], optional
        Random seed for reproducibility. If None, uses current random state.
        Default is None.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - patches1: Tensor of shape (N*P, C, U, V, W) containing patches at (x_i, y_i, z_i)
        - patches2: Tensor of shape (N*P, C, U, V, W) containing patches at (x_i + dx_i, y_i + dy_i, z_i + dz_i)
        - deltas: Tensor of shape (N*P, 3) containing (dx_i, dy_i, dz_i) displacement vectors

    Raises
    ------
    ValueError
        If delta_range constraints are violated or volume dimensions are too small
        for the specified window and delta range.

    Examples
    --------
    >>> tensor = torch.randn(5, 1, 64, 64, 64)  # 5 volumes, 1 channel, 64x64x64
    >>> window = (16, 16, 16)  # 16x16x16 patches
    >>> num_patches = 10  # 10 patch pairs per volume
    >>> delta_range = (8.0, 16.0)  # Euclidean distance between 8 and 16 voxels
    >>> patches1, patches2, deltas = extract_patch_pairs_3d(tensor, window, num_patches, delta_range)
    >>> print(patches1.shape)  # (50, 1, 16, 16, 16)
    >>> print(patches2.shape)  # (50, 1, 16, 16, 16)
    >>> print(deltas.shape)    # (50, 3)
    """
    # Validate input tensor shape
    if len(tensor.shape) != 5:
        raise ValueError(
            f"Input tensor must be 5D (N, C, Z, Y, X), got shape {tensor.shape}"
        )

    N, C, Z, Y, X = tensor.shape
    U, V, W = window

    # Validate delta_range constraints
    max_window = max(U, V, W)
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

    # Check if volume is large enough for window and delta range
    min_z = U + int(high)
    min_y = V + int(high)
    min_x = W + int(high)
    if Z < min_z or Y < min_y or X < min_x:
        raise ValueError(
            f"Volume dimensions ({Z}, {Y}, {X}) are too small for window ({U}, {V}, {W}) "
            f"and delta_range ({low}, {high}). Minimum required: ({min_z}, {min_y}, {min_x})"
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
        (total_patches, C, U, V, W), dtype=tensor.dtype, device=tensor.device
    )
    patches2 = torch.empty(
        (total_patches, C, U, V, W), dtype=tensor.dtype, device=tensor.device
    )
    deltas_tensor = torch.empty(
        (total_patches, 3), dtype=torch.float32, device=tensor.device
    )

    patch_idx = 0

    # Process each volume
    for n in range(N):
        volume = tensor[n]  # Shape: (C, Z, Y, X)

        # Extract P patch pairs for this volume
        for _ in range(num_patches):
            # Sample displacement vector (dx, dy, dz) with Euclidean distance constraint
            dx, dy, dz = _sample_displacement_vector_3d(
                low, high, generator, device=tensor.device
            )

            # Sample first patch location (x, y, z) ensuring both patches fit
            x_min = max(0, -dx)
            x_max = min(X - W, X - W - dx)
            y_min = max(0, -dy)
            y_max = min(Y - V, Y - V - dy)
            z_min = max(0, -dz)
            z_max = min(Z - U, Z - U - dz)

            if x_min >= x_max or y_min >= y_max or z_min >= z_max:
                # If displacement is too large, try again with a smaller one
                attempts = 0
                while (
                    x_min >= x_max or y_min >= y_max or z_min >= z_max
                ) and attempts < 10:
                    dx, dy, dz = _sample_displacement_vector_3d(
                        low, high, generator, device=tensor.device
                    )
                    x_min = max(0, -dx)
                    x_max = min(X - W, X - W - dx)
                    y_min = max(0, -dy)
                    y_max = min(Y - V, Y - V - dy)
                    z_min = max(0, -dz)
                    z_max = min(Z - U, Z - U - dz)
                    attempts += 1

                if x_min >= x_max or y_min >= y_max or z_min >= z_max:
                    raise ValueError(
                        f"Could not find valid patch locations for displacement ({dx}, {dy}, {dz}) "
                        f"in volume of size ({Z}, {Y}, {X}) with window ({U}, {V}, {W})"
                    )

            # Sample random location for first patch
            if generator is not None:
                x = torch.randint(
                    x_min, x_max, (1,), generator=generator, device=tensor.device
                )[0]
                y = torch.randint(
                    y_min, y_max, (1,), generator=generator, device=tensor.device
                )[0]
                z = torch.randint(
                    z_min, z_max, (1,), generator=generator, device=tensor.device
                )[0]
            else:
                x = torch.randint(x_min, x_max, (1,), device=tensor.device)[0]
                y = torch.randint(y_min, y_max, (1,), device=tensor.device)[0]
                z = torch.randint(z_min, z_max, (1,), device=tensor.device)[0]

            # Convert to Python int for slicing
            x_int = int(x)
            y_int = int(y)
            z_int = int(z)

            # Extract first patch at (x, y, z)
            patch1 = volume[
                :, z_int : z_int + U, y_int : y_int + V, x_int : x_int + W
            ]  # Shape: (C, U, V, W)

            # Extract second patch at (x + dx, y + dy, z + dz)
            patch2 = volume[
                :,
                z_int + dz : z_int + dz + U,
                y_int + dy : y_int + dy + V,
                x_int + dx : x_int + dx + W,
            ]  # Shape: (C, U, V, W)

            # Store patches and delta directly in pre-allocated tensors
            patches1[patch_idx] = patch1
            patches2[patch_idx] = patch2
            deltas_tensor[patch_idx, 0] = float(dx)
            deltas_tensor[patch_idx, 1] = float(dy)
            deltas_tensor[patch_idx, 2] = float(dz)

            patch_idx += 1

    return patches1, patches2, deltas_tensor


def _sample_displacement_vector_3d(
    low: float,
    high: float,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, int, int]:
    """
    Sample a displacement vector (dx, dy, dz) such that low <= sqrt(dx² + dy² + dz²) <= high.

    Uses rejection sampling to ensure the Euclidean distance constraint is satisfied.

    Parameters
    ----------
    low : float
        Minimum Euclidean distance
    high : float
        Maximum Euclidean distance
    generator : Optional[torch.Generator]
        Random number generator for reproducibility
    device : Optional[torch.device]
        Device for tensor operations

    Returns
    -------
    Tuple[int, int, int]
        Displacement vector (dx, dy, dz) as integers
    """
    max_attempts = 1000
    for _ in range(max_attempts):
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
            dz_tensor = torch.randint(
                -max_delta, max_delta + 1, (1,), generator=generator, device=device
            )
        else:
            dx_tensor = torch.randint(-max_delta, max_delta + 1, (1,), device=device)
            dy_tensor = torch.randint(-max_delta, max_delta + 1, (1,), device=device)
            dz_tensor = torch.randint(-max_delta, max_delta + 1, (1,), device=device)

        dx = int(dx_tensor[0])
        dy = int(dy_tensor[0])
        dz = int(dz_tensor[0])

        # Check Euclidean distance constraint
        distance = (dx**2 + dy**2 + dz**2) ** 0.5
        if low <= distance <= high:
            return dx, dy, dz

    # If we couldn't find a valid vector after many attempts, use a fallback
    # Sample using spherical coordinates
    if generator is not None:
        theta_tensor = (
            torch.rand(1, generator=generator, device=device) * 2 * 3.141592653589793
        )
        phi_tensor = (
            torch.rand(1, generator=generator, device=device) * 3.141592653589793
        )
        distance_tensor = low + (high - low) * torch.rand(
            1, generator=generator, device=device
        )
    else:
        theta_tensor = torch.rand(1, device=device) * 2 * 3.141592653589793
        phi_tensor = torch.rand(1, device=device) * 3.141592653589793
        distance_tensor = low + (high - low) * torch.rand(1, device=device)

    distance = float(distance_tensor[0])

    # Compute dx, dy, dz from spherical coordinates
    cos_theta = torch.cos(theta_tensor)[0]
    sin_theta = torch.sin(theta_tensor)[0]
    cos_phi = torch.cos(phi_tensor)[0]
    sin_phi = torch.sin(phi_tensor)[0]

    dx = int(round(distance * float(sin_phi) * float(cos_theta)))
    dy = int(round(distance * float(sin_phi) * float(sin_theta)))
    dz = int(round(distance * float(cos_phi)))

    # Ensure distance is still in range (may have been affected by rounding)
    actual_distance = (dx**2 + dy**2 + dz**2) ** 0.5
    if actual_distance < low:
        scale = low / actual_distance
        dx = int(round(dx * scale))
        dy = int(round(dy * scale))
        dz = int(round(dz * scale))
    elif actual_distance > high:
        scale = high / actual_distance
        dx = int(round(dx * scale))
        dy = int(round(dy * scale))
        dz = int(round(dz * scale))

    return dx, dy, dz


def extract_overlapping_pixels_3d(
    patches1: torch.Tensor,
    patches2: torch.Tensor,
    deltas: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract overlapping pixels from 3D patch pairs based on displacement vectors.

    For each patch pair, this function finds pixels that have valid correspondences
    between the two patches (i.e., pixels that represent the same spatial location
    in the original volume). Only overlapping pixels are returned.

    Parameters
    ----------
    patches1 : torch.Tensor
        First set of patches, shape (N*P, C, U, V, W) where:
        - N*P: Total number of patch pairs
        - C: Number of channels
        - U: Patch depth
        - V: Patch height
        - W: Patch width
    patches2 : torch.Tensor
        Second set of patches, shape (N*P, C, U, V, W), corresponding patches
        extracted at displaced locations
    deltas : torch.Tensor
        Displacement vectors, shape (N*P, 3) containing (dx, dy, dz) for each pair

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
    >>> patches1 = torch.randn(10, 1, 16, 16, 16)
    >>> patches2 = torch.randn(10, 1, 16, 16, 16)
    >>> deltas = torch.tensor([[5, 3, 2], [-2, 4, 1], ...])  # shape (10, 3)
    >>> overlapping1, overlapping2 = extract_overlapping_pixels_3d(patches1, patches2, deltas)
    >>> print(overlapping1.shape)  # (K, 1) where K depends on overlap
    >>> print(overlapping2.shape)  # (K, 1)
    >>> # overlapping1[i] and overlapping2[i] correspond to the same spatial location
    """
    # Validate input shapes
    if len(patches1.shape) != 5 or len(patches2.shape) != 5:
        raise ValueError(
            f"Both patches1 and patches2 must be 5D tensors (N*P, C, U, V, W), "
            f"got shapes {patches1.shape} and {patches2.shape}"
        )

    if patches1.shape != patches2.shape:
        raise ValueError(
            f"patches1 and patches2 must have the same shape, "
            f"got {patches1.shape} and {patches2.shape}"
        )

    if len(deltas.shape) != 2 or deltas.shape[1] != 3:
        raise ValueError(
            f"deltas must be 2D tensor of shape (N*P, 3), got {deltas.shape}"
        )

    num_pairs, C, U, V, W = patches1.shape

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
        dz_tensor = deltas_int[i, 2]
        # Convert to Python int only when needed for indexing
        dx = int(dx_tensor)
        dy = int(dy_tensor)
        dz = int(dz_tensor)

        # Get the two patches
        patch1 = patches1[i]  # Shape: (C, U, V, W)
        patch2 = patches2[i]  # Shape: (C, U, V, W)

        # Find valid overlap region in patch1 coordinates
        # A pixel at (u1, v1, w1) in patch1 corresponds to (u1 - dz, v1 - dy, w1 - dx) in patch2
        # For valid correspondence, we need:
        #   0 <= u1 - dz < U  and  0 <= v1 - dy < V  and  0 <= w1 - dx < W
        # Which means: dz <= u1 < U + dz  and  dy <= v1 < V + dy  and  dx <= w1 < W + dx
        # Combined with u1 in [0, U), v1 in [0, V), w1 in [0, W):
        u_min = max(0, dz)
        u_max = min(U, U + dz)
        v_min = max(0, dy)
        v_max = min(V, V + dy)
        w_min = max(0, dx)
        w_max = min(W, W + dx)

        # Check if there's any overlap
        if u_min >= u_max or v_min >= v_max or w_min >= w_max:
            # No overlap for this patch pair, skip it
            continue

        # Extract overlapping region from patch1
        overlap_region1 = patch1[
            :, u_min:u_max, v_min:v_max, w_min:w_max
        ]  # Shape: (C, u_max-u_min, v_max-v_min, w_max-w_min)

        # Extract corresponding region from patch2
        # In patch2 coordinates: u2 = u1 - dz, v2 = v1 - dy, w2 = w1 - dx
        u2_min = u_min - dz
        u2_max = u_max - dz
        v2_min = v_min - dy
        v2_max = v_max - dy
        w2_min = w_min - dx
        w2_max = w_max - dx

        overlap_region2 = patch2[
            :, u2_min:u2_max, v2_min:v2_max, w2_min:w2_max
        ]  # Shape: (C, u_max-u_min, v_max-v_min, w_max-w_min)

        # Reshape to (C, K') where K' is the number of overlapping pixels for this pair
        K_prime = (u_max - u_min) * (v_max - v_min) * (w_max - w_min)
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
