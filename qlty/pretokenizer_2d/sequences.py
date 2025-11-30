"""Pre-tokenization utilities for patch processing.

This module provides functions to prepare patches for tokenization by splitting them
into subpatches (tokens) and computing overlap information between patch pairs.
The actual tokenization (conversion to embeddings) is done by downstream models.
"""

from __future__ import annotations

import torch

from qlty.qlty2D import NCYXQuilt

# Try to import numba for JIT compilation
try:
    import numpy as np
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def tokenize_patch(
    patch: torch.Tensor,
    patch_size: int,
    stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-tokenize a patch by splitting it into a sequence of subpatches with absolute coordinates.

    This is a pre-tokenization step that splits the patch into square subpatches
    (potentially overlapping) using a sliding window approach. The subpatches are
    returned as a sequence with their absolute coordinates within the patch.
    These subpatches can then be tokenized (converted to embeddings) by downstream
    models. Subpatches are extracted such that they never extend beyond patch boundaries.

    This function uses qlty's NCYXQuilt framework for patch extraction, ensuring
    consistency with the rest of the qlty codebase.

    Parameters
    ----------
    patch : torch.Tensor
        Input patch of shape (C, H, W) where:
        - C: Number of channels
        - H: Height of patch
        - W: Width of patch
    patch_size : int
        Size of each token in pixels
    stride : int, optional
        Stride for sliding window extraction. Defaults to patch_size // 2.
        Must be positive and <= patch_size.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - tokens: Tensor of shape (T, C * patch_size * patch_size) where T is
          the number of tokens. Tokens are in row-major order.
        - coords: Tensor of shape (T, 2) containing absolute pixel coordinates
          (y, x) of the top-left corner of each token within the patch.

    Examples
    --------
    >>> patch = torch.randn(3, 16, 16)  # 3 channels, 16x16 patch
    >>> tokens, coords = tokenize_patch(patch, patch_size=4)
    >>> print(tokens.shape)  # (25, 48) - 25 tokens with stride=2, each 3*4*4=48 dims
    >>> print(coords.shape)  # (25, 2) - coordinates for each token
    >>> # coords[0] = [0, 0] for top-left token
    >>> # coords[1] = [0, 2] for next token to the right (with stride=2)
    """
    if len(patch.shape) != 3:
        msg = f"patch must be 3D (C, H, W), got shape {patch.shape}"
        raise ValueError(msg)

    C, H, W = patch.shape

    if patch_size <= 0:
        msg = f"patch_size must be positive, got {patch_size}"
        raise ValueError(msg)

    if stride is None:
        stride = patch_size // 2

    if stride <= 0:
        msg = f"stride must be positive, got {stride}"
        raise ValueError(msg)

    if stride > patch_size:
        msg = f"stride ({stride}) must be <= patch_size ({patch_size})"
        raise ValueError(msg)

    if patch_size > H or patch_size > W:
        msg = f"patch_size ({patch_size}) must be <= patch dimensions ({H}, {W})"
        raise ValueError(
            msg,
        )

    # Use qlty's NCYXQuilt framework for patch extraction
    quilt = NCYXQuilt(
        Y=H,
        X=W,
        window=(patch_size, patch_size),
        step=(stride, stride),
        border=None,  # No border weighting needed for tokenization
    )

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    patch_batch = patch.unsqueeze(0)

    # Extract patches using qlty's unstitch: (1, C, H, W) -> (T, C, patch_size, patch_size)
    patches = quilt.unstitch(patch_batch)

    # Flatten patches: (T, C, patch_size, patch_size) -> (T, C * patch_size * patch_size)
    T = patches.shape[0]
    tokens = patches.view(T, C * patch_size * patch_size)

    # Compute coordinates using the same logic as NCYXQuilt.unstitch()
    # This ensures consistency with how patches are extracted
    coords_list = []
    nY, nX = quilt.get_times()
    for yy in range(nY):
        for xx in range(nX):
            start_y = min(yy * stride, H - patch_size)
            start_x = min(xx * stride, W - patch_size)
            coords_list.append([start_y, start_x])

    coords = torch.tensor(coords_list, dtype=torch.long, device=patch.device)

    return tokens, coords


def _find_overlapping_tokens(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    dx: float,
    dy: float,
    rot_k90: int,
    patch_size: int,
    patch_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find overlapping tokens between two patches given their geometric relationship.

    Parameters
    ----------
    coords1 : torch.Tensor
        Token coordinates from patch1, shape (T1, 2) with (y, x) pixel coordinates
    coords2 : torch.Tensor
        Token coordinates from patch2, shape (T2, 2) with (y, x) pixel coordinates
    dx : float
        Translation in pixels along x-axis
    dy : float
        Translation in pixels along y-axis
    rot_k90 : int
        Rotation applied to patch2 in 90-degree increments (0, 1, 2, or 3)
    patch_size : int
        Size of each token in pixels
    patch_shape : Tuple[int, int]
        Shape of the patch (H, W) in pixels

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - overlap_mask1: Boolean tensor of shape (T1,) indicating which tokens
          in patch1 have overlaps in patch2
        - overlap_indices1_to_2: Long tensor of shape (T1,) where overlap_indices1_to_2[i]
          is the index into coords2/tokens2 for the overlapping token, or -1
          if no overlap exists
        - overlap_mask2: Boolean tensor of shape (T2,) indicating which tokens
          in patch2 have overlaps in patch1
        - overlap_indices2_to_1: Long tensor of shape (T2,) where overlap_indices2_to_1[j]
          is the index into coords1/tokens1 for the overlapping token, or -1
          if no overlap exists
        - overlap_fractions: Float tensor of shape (T1,) containing the fraction
          of overlap for each token in patch1 (0.0 to 1.0), or 0.0 if no overlap
    """
    T1 = coords1.shape[0]
    T2 = coords2.shape[0]
    H, W = patch_shape
    rot_k90 = rot_k90 % 4

    overlap_mask1 = torch.zeros(T1, dtype=torch.bool, device=coords1.device)
    overlap_indices1_to_2 = torch.full(
        (T1,),
        -1,
        dtype=torch.long,
        device=coords1.device,
    )
    overlap_mask2 = torch.zeros(T2, dtype=torch.bool, device=coords2.device)
    overlap_indices2_to_1 = torch.full(
        (T2,),
        -1,
        dtype=torch.long,
        device=coords2.device,
    )
    overlap_fractions = torch.zeros(T1, dtype=torch.float32, device=coords1.device)

    token_area = float(patch_size * patch_size)

    # For each token in patch1, find if it overlaps with any token in patch2
    for i in range(T1):
        y1 = coords1[i, 0].item()
        x1 = coords1[i, 1].item()

        # Token region in patch1: [y1, y1+patch_size) x [x1, x1+patch_size)
        # Transform the four corners of this token to patch2's coordinate system
        corners1 = [
            (x1, y1),  # top-left
            (x1 + patch_size, y1),  # top-right
            (x1, y1 + patch_size),  # bottom-left
            (x1 + patch_size, y1 + patch_size),  # bottom-right
        ]

        # Transform corners to patch2's coordinate system
        corners2 = []
        for x, y in corners1:
            # Apply inverse translation
            x_unrot = x - dx
            y_unrot = y - dy

            # Apply inverse rotation
            if rot_k90 == 0:
                x2 = x_unrot
                y2 = y_unrot
            elif rot_k90 == 1:
                # 90° clockwise rotation: (x, y) -> (y, W - x)
                # Inverse: (x, y) -> (W - y, x)
                x2 = W - y_unrot
                y2 = x_unrot
            elif rot_k90 == 2:
                # 180° rotation: (x, y) -> (W - x, H - y)
                x2 = W - x_unrot
                y2 = H - y_unrot
            elif rot_k90 == 3:
                # 270° clockwise: (x, y) -> (H - y, x)
                # Inverse: (x, y) -> (y, H - x)
                x2 = y_unrot
                y2 = H - x_unrot
            else:
                msg = f"Invalid rotation: {rot_k90}"
                raise ValueError(msg)

            corners2.append((x2, y2))

        # Find bounding box of transformed token in patch2
        x2_min = min(x for x, y in corners2)
        x2_max = max(x for x, y in corners2)
        y2_min = min(y for x, y in corners2)
        y2_max = max(y for x, y in corners2)

        # Check which tokens in patch2 overlap with this bounding box
        best_overlap = 0.0
        best_j = -1

        for j in range(coords2.shape[0]):
            y2 = coords2[j, 0].item()
            x2 = coords2[j, 1].item()

            # Token region in patch2: [y2, y2+patch_size) x [x2, x2+patch_size)
            # Compute intersection with transformed token from patch1
            # Intersection in patch2 coordinates
            intersect_x_min = max(x2_min, x2)
            intersect_x_max = min(x2_max, x2 + patch_size)
            intersect_y_min = max(y2_min, y2)
            intersect_y_max = min(y2_max, y2 + patch_size)

            if intersect_x_min < intersect_x_max and intersect_y_min < intersect_y_max:
                # There is an intersection
                # Transform intersection back to patch1 coordinates to compute area
                # For simplicity, approximate using the intersection in patch2
                # (this is exact for integer translations and rotations)
                intersect_area = (intersect_x_max - intersect_x_min) * (
                    intersect_y_max - intersect_y_min
                )
                overlap_frac = intersect_area / token_area

                if overlap_frac > best_overlap:
                    best_overlap = overlap_frac
                    best_j = j

        if best_overlap > 0.0:
            overlap_mask1[i] = True
            overlap_indices1_to_2[i] = best_j
            overlap_fractions[i] = best_overlap
            # Also set reverse mapping (use same fraction since tokens are same size)
            if not overlap_mask2[best_j]:
                overlap_mask2[best_j] = True
                overlap_indices2_to_1[best_j] = i

    return (
        overlap_mask1,
        overlap_indices1_to_2,
        overlap_mask2,
        overlap_indices2_to_1,
        overlap_fractions,
    )


if HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _numba_find_overlaps_batch(
        coords: np.ndarray,  # (T, 2) float64
        dx: np.ndarray,  # (N,) float32
        dy: np.ndarray,  # (N,) float32
        rot_k90: np.ndarray,  # (N,) int64
        patch_size: int,
        H: int,
        W: int,
        overlap_mask1_out: np.ndarray,  # (N, T) bool
        overlap_indices1_to_2_out: np.ndarray,  # (N, T) int64
        overlap_mask2_out: np.ndarray,  # (N, T) bool
        overlap_indices2_to_1_out: np.ndarray,  # (N, T) int64
        overlap_fractions_out: np.ndarray,  # (N, T) float32
    ):
        """
        Numba-accelerated batch overlap computation.

        Processes all pairs in parallel using prange.
        """
        N = dx.shape[0]
        T = coords.shape[0]
        token_area = float(patch_size * patch_size)

        # Process each patch pair in parallel
        for n in prange(N):
            dx_val = dx[n]
            dy_val = dy[n]
            rot = int(rot_k90[n]) % 4

            # For each token in patch1, find best overlap in patch2
            for i in range(T):
                y1 = coords[i, 0]
                x1 = coords[i, 1]

                # Transform four corners to patch2's coordinate system
                # Corner coordinates: top-left, top-right, bottom-left, bottom-right
                x1_tl = x1
                y1_tl = y1
                x1_tr = x1 + patch_size
                y1_tr = y1
                x1_bl = x1
                y1_bl = y1 + patch_size
                x1_br = x1 + patch_size
                y1_br = y1 + patch_size

                # Apply inverse translation and rotation to each corner
                # Transform corner 1 (top-left)
                x_unrot_tl = x1_tl - dx_val
                y_unrot_tl = y1_tl - dy_val
                if rot == 0:
                    x2_tl, y2_tl = x_unrot_tl, y_unrot_tl
                elif rot == 1:
                    x2_tl, y2_tl = W - y_unrot_tl, x_unrot_tl
                elif rot == 2:
                    x2_tl, y2_tl = W - x_unrot_tl, H - y_unrot_tl
                elif rot == 3:
                    x2_tl, y2_tl = y_unrot_tl, H - x_unrot_tl
                else:
                    x2_tl, y2_tl = x_unrot_tl, y_unrot_tl

                # Transform corner 2 (top-right)
                x_unrot_tr = x1_tr - dx_val
                y_unrot_tr = y1_tr - dy_val
                if rot == 0:
                    x2_tr, y2_tr = x_unrot_tr, y_unrot_tr
                elif rot == 1:
                    x2_tr, y2_tr = W - y_unrot_tr, x_unrot_tr
                elif rot == 2:
                    x2_tr, y2_tr = W - x_unrot_tr, H - y_unrot_tr
                elif rot == 3:
                    x2_tr, y2_tr = y_unrot_tr, H - x_unrot_tr
                else:
                    x2_tr, y2_tr = x_unrot_tr, y_unrot_tr

                # Transform corner 3 (bottom-left)
                x_unrot_bl = x1_bl - dx_val
                y_unrot_bl = y1_bl - dy_val
                if rot == 0:
                    x2_bl, y2_bl = x_unrot_bl, y_unrot_bl
                elif rot == 1:
                    x2_bl, y2_bl = W - y_unrot_bl, x_unrot_bl
                elif rot == 2:
                    x2_bl, y2_bl = W - x_unrot_bl, H - y_unrot_bl
                elif rot == 3:
                    x2_bl, y2_bl = y_unrot_bl, H - x_unrot_bl
                else:
                    x2_bl, y2_bl = x_unrot_bl, y_unrot_bl

                # Transform corner 4 (bottom-right)
                x_unrot_br = x1_br - dx_val
                y_unrot_br = y1_br - dy_val
                if rot == 0:
                    x2_br, y2_br = x_unrot_br, y_unrot_br
                elif rot == 1:
                    x2_br, y2_br = W - y_unrot_br, x_unrot_br
                elif rot == 2:
                    x2_br, y2_br = W - x_unrot_br, H - y_unrot_br
                elif rot == 3:
                    x2_br, y2_br = y_unrot_br, H - x_unrot_br
                else:
                    x2_br, y2_br = x_unrot_br, y_unrot_br

                # Find bounding box
                x2_min = min(x2_tl, x2_tr, x2_bl, x2_br)
                x2_max = max(x2_tl, x2_tr, x2_bl, x2_br)
                y2_min = min(y2_tl, y2_tr, y2_bl, y2_br)
                y2_max = max(y2_tl, y2_tr, y2_bl, y2_br)

                # Find best overlap with tokens in patch2
                best_overlap = 0.0
                best_j = -1

                for j in range(T):
                    y2 = coords[j, 0]
                    x2 = coords[j, 1]

                    # Compute intersection
                    intersect_x_min = max(x2_min, x2)
                    intersect_x_max = min(x2_max, x2 + patch_size)
                    intersect_y_min = max(y2_min, y2)
                    intersect_y_max = min(y2_max, y2 + patch_size)

                    if (
                        intersect_x_min < intersect_x_max
                        and intersect_y_min < intersect_y_max
                    ):
                        intersect_area = (intersect_x_max - intersect_x_min) * (
                            intersect_y_max - intersect_y_min
                        )
                        overlap_frac = intersect_area / token_area

                        if overlap_frac > best_overlap:
                            best_overlap = overlap_frac
                            best_j = j

                # Store results
                if best_overlap > 0.0:
                    overlap_mask1_out[n, i] = True
                    overlap_indices1_to_2_out[n, i] = best_j
                    overlap_fractions_out[n, i] = best_overlap
                    # Set reverse mapping (only if not already set by another token)
                    if not overlap_mask2_out[n, best_j]:
                        overlap_mask2_out[n, best_j] = True
                        overlap_indices2_to_1_out[n, best_j] = i


def _to_tensor_batch(
    value,
    N: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """
    Convert a value (scalar, tensor, or numpy array) to a batched tensor.

    Parameters
    ----------
    value : scalar, torch.Tensor, or numpy.ndarray
        Input value to convert
    N : int
        Batch size
    dtype : torch.dtype
        Target dtype
    device : torch.device
        Target device
    name : str
        Name of parameter (for error messages)

    Returns
    -------
    torch.Tensor
        Tensor of shape (N,) on the specified device
    """
    if isinstance(value, torch.Tensor):
        value = value.to(device)
        if value.shape[0] != N:
            msg = f"{name} must have shape ({N},) or be scalar, got {value.shape}"
            raise ValueError(
                msg,
            )
        return value

    # Handle numpy arrays or scalars
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value).to(device)
            if value.shape[0] != N:
                msg = f"{name} must have shape ({N},) or be scalar, got {value.shape}"
                raise ValueError(
                    msg,
                )
            return value
    except ImportError:
        pass

    # Scalar: broadcast to batch
    return torch.tensor([value] * N, dtype=dtype, device=device)


def build_sequence_pair(
    patch1: torch.Tensor,
    patch2: torch.Tensor,
    dx: float | torch.Tensor,
    dy: float | torch.Tensor,
    rot_k90: int | torch.Tensor,
    patch_size: int,
    stride: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Build sequence pair from two patches with overlap information.

    This function pre-tokenizes both patches (splits them into subpatches), finds
    overlapping subpatches, and returns sequences with absolute coordinates suitable
    for downstream tokenization and embedding methods.

    Supports both single patches and batches:
    - Single: patch1/patch2 shape (C, H, W), dx/dy/rot_k90 are scalars
    - Batch: patch1/patch2 shape (N, C, H, W), dx/dy/rot_k90 are (N,) tensors

    Parameters
    ----------
    patch1 : torch.Tensor
        First patch of shape (C, H, W) or batch of shape (N, C, H, W)
    patch2 : torch.Tensor
        Second patch of shape (C, H, W) or batch of shape (N, C, H, W)
    dx : float or torch.Tensor
        Translation in pixels along x-axis. Scalar for single patch, (N,) tensor for batch.
    dy : float or torch.Tensor
        Translation in pixels along y-axis. Scalar for single patch, (N,) tensor for batch.
    rot_k90 : int or torch.Tensor
        Rotation applied to patch2 in 90-degree increments (0, 1, 2, or 3).
        Scalar for single patch, (N,) tensor for batch.
    patch_size : int
        Size of each token in pixels
    stride : int, optional
        Stride for sliding window token extraction. Defaults to patch_size // 2.
        Must be positive and <= patch_size.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing:

        For single patch:
        - "tokens1": Token vectors from patch1, shape (T1, D)
        - "tokens2": Token vectors from patch2, shape (T2, D)
        - "coords1": Absolute pixel coordinates (y, x) for patch1 tokens, shape (T1, 2)
        - "coords2": Absolute pixel coordinates (y, x) for patch2 tokens, shape (T2, 2)
        - "overlap_mask1": Boolean mask indicating which patch1 tokens overlap, shape (T1,)
        - "overlap_mask2": Boolean mask indicating which patch2 tokens overlap, shape (T2,)
        - "overlap_indices1_to_2": Mapping from patch1 to patch2 tokens, shape (T1,)
        - "overlap_indices2_to_1": Mapping from patch2 to patch1 tokens, shape (T2,)
        - "overlap_fractions": Fraction of overlap for each patch1 token (0.0 to 1.0), shape (T1,)
        - "overlap_pairs": Tensor of shape (N_overlaps, 2) containing (i, j) pairs

        For batch (all tensors are padded to max length):
        - "tokens1": Token vectors from patch1, shape (N, T_max, D)
        - "tokens2": Token vectors from patch2, shape (N, T_max, D)
        - "coords1": Absolute pixel coordinates for patch1 tokens, shape (N, T_max, 2)
        - "coords2": Absolute pixel coordinates for patch2 tokens, shape (N, T_max, 2)
        - "overlap_mask1": Boolean mask for patch1 tokens, shape (N, T_max)
        - "overlap_mask2": Boolean mask for patch2 tokens, shape (N, T_max)
        - "overlap_indices1_to_2": Mapping from patch1 to patch2, shape (N, T_max), -1 for padding
        - "overlap_indices2_to_1": Mapping from patch2 to patch1, shape (N, T_max), -1 for padding
        - "overlap_fractions": Fraction of overlap, shape (N, T_max)
        - "overlap_pairs": Overlap pairs, shape (N, max_pairs, 2), -1 for padding
        - "sequence_lengths": Actual sequence lengths (same for both patches), shape (N,)
        - "overlap_pair_counts": Number of overlap pairs per sample, shape (N,)
    """
    # Check if inputs are batched
    is_batched = len(patch1.shape) == 4 and len(patch2.shape) == 4

    if is_batched:
        # Batch processing
        N, C1, H1, W1 = patch1.shape
        N2, C2, H2, W2 = patch2.shape

        if N != N2:
            msg = f"Batch sizes must match: patch1 has {N} patches, patch2 has {N2}"
            raise ValueError(
                msg,
            )

        if C1 != C2 or H1 != H2 or W1 != W2:
            msg = f"Patches must have same shape, got {patch1.shape} and {patch2.shape}"
            raise ValueError(
                msg,
            )

        # Convert dx, dy, rot_k90 to tensors if needed
        dx = _to_tensor_batch(dx, N, torch.float32, patch1.device, "dx")
        dy = _to_tensor_batch(dy, N, torch.float32, patch1.device, "dy")
        rot_k90 = _to_tensor_batch(rot_k90, N, torch.int64, patch1.device, "rot_k90")

        # OPTIMIZATION: Batch tokenize all patches at once
        # Since all patches have the same shape, we can use a single quilt object
        C, H, W = patch1.shape[1:]

        # Determine stride (same logic as tokenize_patch)
        stride_val = patch_size // 2 if stride is None else stride

        # Create quilt object once (same for all patches)
        quilt = NCYXQuilt(
            Y=H,
            X=W,
            window=(patch_size, patch_size),
            step=(stride_val, stride_val),
            border=None,
        )

        # Batch tokenize: (N, C, H, W) -> (N*T, C, patch_size, patch_size)
        patches1_flat = quilt.unstitch(patch1)  # (N*T, C, patch_size, patch_size)
        patches2_flat = quilt.unstitch(patch2)  # (N*T, C, patch_size, patch_size)

        # Get number of tokens per patch
        nY, nX = quilt.get_times()
        T = nY * nX  # Same for all patches

        # Compute coordinates once (same for all patches)
        # Use the same logic as NCYXQuilt.unstitch() to ensure consistency
        coords_list = []
        for yy in range(nY):
            for xx in range(nX):
                start_y = min(yy * stride_val, H - patch_size)
                start_x = min(xx * stride_val, W - patch_size)
                coords_list.append([start_y, start_x])
        coords = torch.tensor(
            coords_list,
            dtype=torch.long,
            device=patch1.device,
        )  # (T, 2)

        # Flatten patches to tokens: (N*T, C, patch_size, patch_size) -> (N*T, C*patch_size*patch_size)
        D = C * patch_size * patch_size
        tokens1_flat = patches1_flat.view(N * T, D)  # (N*T, D)
        tokens2_flat = patches2_flat.view(N * T, D)  # (N*T, D)

        # Reshape to (N, T, D)
        tokens1_batch = tokens1_flat.view(N, T, D)
        tokens2_batch = tokens2_flat.view(N, T, D)

        # Expand coordinates for all patches: (T, 2) -> (N, T, 2)
        coords1_batch = coords.unsqueeze(0).expand(N, -1, -1)
        coords2_batch = coords.unsqueeze(0).expand(N, -1, -1)

        # Initialize overlap tensors
        overlap_mask1_batch = torch.zeros(
            (N, T),
            dtype=torch.bool,
            device=patch1.device,
        )
        overlap_mask2_batch = torch.zeros(
            (N, T),
            dtype=torch.bool,
            device=patch1.device,
        )
        overlap_indices1_to_2_batch = torch.full(
            (N, T),
            -1,
            dtype=torch.long,
            device=patch1.device,
        )
        overlap_indices2_to_1_batch = torch.full(
            (N, T),
            -1,
            dtype=torch.long,
            device=patch1.device,
        )
        overlap_fractions_batch = torch.zeros(
            (N, T),
            dtype=torch.float32,
            device=patch1.device,
        )

        # Process overlaps - use numba-accelerated batch processing if available
        # Otherwise fall back to sequential or threading
        use_numba = HAS_NUMBA and N > 5  # Use numba for batches larger than 5

        if use_numba:
            # Use numba-accelerated batch processing with parallel execution
            # Convert tensors to numpy for numba
            coords_np = coords.cpu().numpy().astype(np.float64)
            dx_np = dx.cpu().numpy().astype(np.float32)
            dy_np = dy.cpu().numpy().astype(np.float32)
            rot_k90_np = rot_k90.cpu().numpy().astype(np.int64)

            # Initialize output arrays
            overlap_mask1_np = np.zeros((N, T), dtype=np.bool_)
            overlap_indices1_to_2_np = np.full((N, T), -1, dtype=np.int64)
            overlap_mask2_np = np.zeros((N, T), dtype=np.bool_)
            overlap_indices2_to_1_np = np.full((N, T), -1, dtype=np.int64)
            overlap_fractions_np = np.zeros((N, T), dtype=np.float32)

            # Run numba-accelerated batch computation
            _numba_find_overlaps_batch(
                coords_np,
                dx_np,
                dy_np,
                rot_k90_np,
                patch_size,
                H,
                W,
                overlap_mask1_np,
                overlap_indices1_to_2_np,
                overlap_mask2_np,
                overlap_indices2_to_1_np,
                overlap_fractions_np,
            )

            # Convert back to PyTorch tensors on the original device
            overlap_mask1_batch = torch.from_numpy(overlap_mask1_np).to(patch1.device)
            overlap_mask2_batch = torch.from_numpy(overlap_mask2_np).to(patch1.device)
            overlap_indices1_to_2_batch = torch.from_numpy(overlap_indices1_to_2_np).to(
                patch1.device,
            )
            overlap_indices2_to_1_batch = torch.from_numpy(overlap_indices2_to_1_np).to(
                patch1.device,
            )
            overlap_fractions_batch = torch.from_numpy(overlap_fractions_np).to(
                patch1.device,
            )

            # Build overlap pairs (vectorized for all pairs)
            overlap_pairs_all = []
            for i in range(N):
                mask = overlap_mask1_batch[i]
                if mask.any():
                    indices1 = torch.arange(T, device=patch1.device)[mask]
                    indices2 = overlap_indices1_to_2_batch[i][mask]
                    pairs = torch.stack([indices1, indices2], dim=1)
                    overlap_pairs_all.append(pairs)
                else:
                    overlap_pairs_all.append(
                        torch.empty((0, 2), dtype=torch.long, device=patch1.device),
                    )
        else:
            # Sequential processing (for small batches or when numba unavailable)
            overlap_pairs_all = []
            for i in range(N):
                (
                    overlap_mask1,
                    overlap_indices1_to_2,
                    overlap_mask2,
                    overlap_indices2_to_1,
                    overlap_fractions,
                ) = _find_overlapping_tokens(
                    coords,
                    coords,
                    dx[i].item(),
                    dy[i].item(),
                    rot_k90[i].item(),
                    patch_size,
                    (H, W),
                )

                overlap_mask1_batch[i] = overlap_mask1
                overlap_mask2_batch[i] = overlap_mask2
                overlap_indices1_to_2_batch[i] = overlap_indices1_to_2
                overlap_indices2_to_1_batch[i] = overlap_indices2_to_1
                overlap_fractions_batch[i] = overlap_fractions

                # Build overlap pairs (vectorized)
                mask = overlap_mask1
                if mask.any():
                    indices1 = torch.arange(T, device=patch1.device)[mask]
                    indices2 = overlap_indices1_to_2[mask]
                    pairs = torch.stack([indices1, indices2], dim=1)
                    overlap_pairs_all.append(pairs)
                else:
                    overlap_pairs_all.append(
                        torch.empty((0, 2), dtype=torch.long, device=patch1.device),
                    )

        # Find maximum number of overlap pairs
        max_pairs = (
            max(pairs.shape[0] for pairs in overlap_pairs_all)
            if overlap_pairs_all
            else 0
        )

        # Create overlap_pairs_batch tensor
        if max_pairs == 0:
            overlap_pairs_batch = torch.empty(
                (N, 0, 2),
                dtype=torch.long,
                device=patch1.device,
            )
        else:
            overlap_pairs_batch = torch.full(
                (N, max_pairs, 2),
                -1,
                dtype=torch.long,
                device=patch1.device,
            )
            for i, pairs in enumerate(overlap_pairs_all):
                num_pairs = pairs.shape[0]
                if num_pairs > 0:
                    overlap_pairs_batch[i, :num_pairs] = pairs

        # Create sequence lengths and pair counts
        sequence_lengths = torch.full((N,), T, dtype=torch.long, device=patch1.device)
        overlap_pair_counts = torch.tensor(
            [pairs.shape[0] for pairs in overlap_pairs_all],
            dtype=torch.long,
            device=patch1.device,
        )

        return {
            "tokens1": tokens1_batch,  # (N, T_max, D)
            "tokens2": tokens2_batch,  # (N, T_max, D)
            "coords1": coords1_batch,  # (N, T_max, 2)
            "coords2": coords2_batch,  # (N, T_max, 2)
            "overlap_mask1": overlap_mask1_batch,  # (N, T_max)
            "overlap_mask2": overlap_mask2_batch,  # (N, T_max)
            "overlap_indices1_to_2": overlap_indices1_to_2_batch,  # (N, T_max)
            "overlap_indices2_to_1": overlap_indices2_to_1_batch,  # (N, T_max)
            "overlap_fractions": overlap_fractions_batch,  # (N, T_max)
            "overlap_pairs": overlap_pairs_batch,  # (N, max_pairs, 2)
            "sequence_lengths": sequence_lengths,  # (N,) - actual sequence lengths (same for both patches)
            "overlap_pair_counts": overlap_pair_counts,  # (N,) - number of overlap pairs per sample
        }

    # Single patch processing (original behavior)
    if len(patch1.shape) != 3 or len(patch2.shape) != 3:
        msg = (
            f"Both patches must be 3D (C, H, W) or 4D (N, C, H, W), "
            f"got shapes {patch1.shape} and {patch2.shape}"
        )
        raise ValueError(
            msg,
        )

    C1, H1, W1 = patch1.shape
    C2, H2, W2 = patch2.shape

    if C1 != C2 or H1 != H2 or W1 != W2:
        msg = f"Patches must have same shape, got {patch1.shape} and {patch2.shape}"
        raise ValueError(
            msg,
        )

    # Convert scalars to floats/ints if they're tensors
    if isinstance(dx, torch.Tensor):
        dx = dx.item()
    if isinstance(dy, torch.Tensor):
        dy = dy.item()
    if isinstance(rot_k90, torch.Tensor):
        rot_k90 = rot_k90.item()

    # Tokenize both patches
    tokens1, coords1 = tokenize_patch(patch1, patch_size, stride=stride)
    tokens2, coords2 = tokenize_patch(patch2, patch_size, stride=stride)

    # Find overlapping tokens
    (
        overlap_mask1,
        overlap_indices1_to_2,
        overlap_mask2,
        overlap_indices2_to_1,
        overlap_fractions,
    ) = _find_overlapping_tokens(
        coords1,
        coords2,
        dx,
        dy,
        rot_k90,
        patch_size,
        (H1, W1),
    )

    # Build list of overlap pairs: [(i, j), ...] where token i in patch1 overlaps with token j in patch2
    overlap_pairs = []
    for i in range(overlap_mask1.shape[0]):
        if overlap_mask1[i]:
            j = overlap_indices1_to_2[i].item()
            overlap_pairs.append((i, j))
    overlap_pairs_tensor = (
        torch.tensor(overlap_pairs, dtype=torch.long, device=tokens1.device)
        if overlap_pairs
        else torch.empty((0, 2), dtype=torch.long, device=tokens1.device)
    )

    return {
        "tokens1": tokens1,
        "tokens2": tokens2,
        "coords1": coords1,
        "coords2": coords2,
        "overlap_mask1": overlap_mask1,
        "overlap_mask2": overlap_mask2,
        "overlap_indices1_to_2": overlap_indices1_to_2,
        "overlap_indices2_to_1": overlap_indices2_to_1,
        "overlap_fractions": overlap_fractions,  # Fraction of overlap for each patch1 token (0.0 to 1.0)
        "overlap_pairs": overlap_pairs_tensor,  # Shape (N_overlaps, 2) with (i, j) pairs
    }
