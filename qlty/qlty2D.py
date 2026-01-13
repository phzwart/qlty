from __future__ import annotations

import math

import einops
import numpy as np
import torch
from numba import njit, prange

from qlty.base import (
    compute_border_tensor_torch,
    compute_chunk_times,
    compute_weight_matrix_torch,
    normalize_border,
    validate_border_weight,
)


@njit(fastmath=True)  # pragma: no cover
def numba_njit_stitch(
    ml_tensor,
    result,
    norma,
    weight,
    window,
    step,
    Y,
    X,
    nX,
    times,
    m,
):
    # NOTE:
    # We intentionally avoid `parallel=True` because concurrent updates to
    # shared output slices (`result` and `norma`) introduce race conditions
    # that break test expectations. Keeping the loop serial preserves correctness.
    for i in range(times):
        yy = i // nX
        xx = i % nX
        here_and_now = times * m + yy * nX + xx
        start_y = min(yy * step[0], Y - window[0])
        start_x = min(xx * step[1], X - window[1])
        stop_y = start_y + window[0]
        stop_x = start_x + window[1]
        for j in range(ml_tensor.shape[1]):
            tmp = ml_tensor[here_and_now, j, ...]
            result[m, j, start_y:stop_y, start_x:stop_x] += tmp * weight
        # get the weight matrix, only compute once
        if m == 0:
            norma[start_y:stop_y, start_x:stop_x] += weight
    return result, norma


@njit(fastmath=True, parallel=True)  # pragma: no cover
def numba_njit_stitch_color(
    ml_tensor,
    result,
    norma,
    weight,
    window,
    step,
    Y,
    X,
    nX,
    times,
    m,
    color_y_mod,
    color_x_mod,
    color_y_idx,
    color_x_idx,
):
    for i in prange(times):
        yy = i // nX
        xx = i % nX
        if yy % color_y_mod != color_y_idx or xx % color_x_mod != color_x_idx:
            continue
        here_and_now = times * m + yy * nX + xx
        start_y = min(yy * step[0], Y - window[0])
        start_x = min(xx * step[1], X - window[1])
        stop_y = start_y + window[0]
        stop_x = start_x + window[1]
        for j in range(ml_tensor.shape[1]):
            tmp = ml_tensor[here_and_now, j, ...]
            result[m, j, start_y:stop_y, start_x:stop_x] += tmp * weight
        if m == 0:
            norma[start_y:stop_y, start_x:stop_x] += weight
    return result, norma


def _ensure_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().contiguous().numpy()
    return array


def stitch_serial_numba(
    ml_tensor: torch.Tensor,
    weight: torch.Tensor,
    window: tuple[int, int],
    step: tuple[int, int],
    Y: int,
    X: int,
    nY: int,
    nX: int,
):
    times = nY * nX
    ml_tensor_np = _ensure_numpy(ml_tensor)
    weight_np = _ensure_numpy(weight)

    M_images = ml_tensor_np.shape[0] // times
    assert ml_tensor_np.shape[0] % times == 0

    result_np = np.zeros(
        (M_images, ml_tensor_np.shape[1], Y, X),
        dtype=ml_tensor_np.dtype,
    )
    norma_np = np.zeros((Y, X), dtype=weight_np.dtype)

    for m in range(M_images):
        result_np, norma_np = numba_njit_stitch(
            ml_tensor_np,
            result_np,
            norma_np,
            weight_np,
            window,
            step,
            Y,
            X,
            nX,
            times,
            m,
        )

    result = torch.from_numpy(result_np)
    norma = torch.from_numpy(norma_np)
    result = result / norma
    return result, norma


def stitch_parallel_colored(
    ml_tensor: torch.Tensor,
    weight: torch.Tensor,
    window: tuple[int, int],
    step: tuple[int, int],
    Y: int,
    X: int,
    nY: int,
    nX: int,
):
    times = nY * nX
    ml_tensor_np = _ensure_numpy(ml_tensor)
    weight_np = _ensure_numpy(weight)

    M_images = ml_tensor_np.shape[0] // times
    assert ml_tensor_np.shape[0] % times == 0

    result_np = np.zeros(
        (M_images, ml_tensor_np.shape[1], Y, X),
        dtype=ml_tensor_np.dtype,
    )
    norma_np = np.zeros((Y, X), dtype=weight_np.dtype)

    color_y_mod = 1
    color_x_mod = 1
    if step[0] > 0:
        color_y_mod = max(1, math.ceil(window[0] / step[0]))
    if step[1] > 0:
        color_x_mod = max(1, math.ceil(window[1] / step[1]))
    if nY > 0:
        color_y_mod = min(color_y_mod, nY)
    if nX > 0:
        color_x_mod = min(color_x_mod, nX)

    for m in range(M_images):
        for color_y_idx in range(color_y_mod):
            for color_x_idx in range(color_x_mod):
                result_np, norma_np = numba_njit_stitch_color(
                    ml_tensor_np,
                    result_np,
                    norma_np,
                    weight_np,
                    window,
                    step,
                    Y,
                    X,
                    nX,
                    times,
                    m,
                    color_y_mod,
                    color_x_mod,
                    color_y_idx,
                    color_x_idx,
                )

    result = torch.from_numpy(result_np)
    norma = torch.from_numpy(norma_np)
    result = result / norma
    return result, norma


class NCYXQuilt:
    """
    This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
    This class is aimed at handling tensors of type (N,C,Y,X)

    """

    def __init__(
        self,
        Y: int,
        X: int,
        window: tuple[int, int],
        step: tuple[int, int],
        border: int | tuple[int, int] | None,
        border_weight: float = 1.0,
    ) -> None:
        """
        This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
        This class is aimed at handling tensors of type (N,C,Y,X).

        Parameters
        ----------
        Y : number of elements in the Y direction
        X : number of elements in the X direction
        window: The size of the sliding window, a tuple (Ysub, Xsub)
        step: The step size at which we want to sample the sliding window (Ystep,Xstep)
        border: Border pixels of the window we want to 'ignore' or down weight when stitching things back
        border_weight: The weight for the border pixels, should be between 0 and 1. The default of 0.1 should be fine
        """
        self.Y = Y
        self.X = X
        self.window = window
        self.step = step

        # Normalize and validate border
        self.border = normalize_border(border, ndim=2)
        self.border_weight = validate_border_weight(border_weight)

        # Compute chunk times
        self.nY, self.nX = compute_chunk_times(
            dimension_sizes=(Y, X),
            window=window,
            step=step,
        )

        # Compute weight matrix
        self.weight = compute_weight_matrix_torch(
            window=window,
            border=self.border,
            border_weight=self.border_weight,
        )

    def border_tensor(self) -> torch.Tensor:
        """Compute border tensor indicating valid (non-border) regions."""
        return compute_border_tensor_torch(window=self.window, border=self.border)

    def get_times(self) -> tuple[int, int]:
        """
        Compute the number of patches along each spatial dimension.

        This method calculates how many patches will be created in the Y and X
        dimensions, ensuring the last patch always fits within the image bounds.

        Returns
        -------
        Tuple[int, int]
            A tuple (nY, nX) where:
            - nY: Number of patches in the Y (height) dimension
            - nX: Number of patches in the X (width) dimension

        The total number of patches per image is nY * nX.

        Examples
        --------
        >>> quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16))
        >>> nY, nX = quilt.get_times()
        >>> print(f"Patches per image: {nY * nX}")
        >>> print(f"Total patches for 10 images: {10 * nY * nX}")
        """
        return compute_chunk_times(
            dimension_sizes=(self.Y, self.X),
            window=self.window,
            step=self.step,
        )

    def unstitch_data_pair(
        self,
        tensor_in: torch.Tensor,
        tensor_out: torch.Tensor,
        missing_label: float | None = None,
        return_positions: bool = False,
        include_n_position: bool = False,
        add_positional_channels: bool = False,
        normalize_positions: bool = True,
        position_mode: str = "absolute",
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Split input and output tensors into smaller overlapping patches.

        This method is useful for training neural networks where you need to process
        input-output pairs together. The output tensor can optionally have missing
        labels that will be masked in border regions.

        Parameters
        ----------
        tensor_in : torch.Tensor
            Input tensor of shape (N, C, Y, X). The tensor going into the network.
        tensor_out : torch.Tensor
            Output tensor of shape (N, C, Y, X) or (N, Y, X). The target tensor.
            If 3D, will be automatically expanded to 4D.
        missing_label : Optional[Union[int, float]], optional
            Label value that indicates missing/invalid data. If provided, pixels
            in the border region will be set to this value in the output patches.
            Default is None (no masking).
        return_positions : bool, optional
            If True, also return positional embeddings. Default is False.
        include_n_position : bool, optional
            If True and return_positions=True, include N (batch) index in positions.
            If False, positions only contain [Y_pos, X_pos]. Default is False.
        add_positional_channels : bool, optional
            If True, add positional embedding channels directly to input patches.
            Adds 2 channels (Y and X coordinates) to each patch.
            Default is False.
        normalize_positions : bool, optional
            If True and add_positional_channels=True, normalize coordinates to [0, 1].
            If False, use raw pixel coordinates. Default is True.
        position_mode : str, optional
            Mode for positional channels when add_positional_channels=True:
            - "absolute": Use absolute image coordinates (offset by patch position)
            - "relative": Use relative coordinates within patch (0 to window-1)
            Default is "absolute".

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            If return_positions=False:
            - input_patches: Shape (M, C, window[0], window[1]) or (M, C+2, window[0], window[1]) if add_positional_channels=True
            - output_patches: Shape (M, C, window[0], window[1]) or (M, window[0], window[1])
            where M = N * nY * nX

            If return_positions=True, additionally returns:
            - positions: Tensor of shape (M, 2) or (M, 3) containing [Y_pos, X_pos] or [N_idx, Y_pos, X_pos]

        Examples
        --------
        >>> quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5))
        >>> input_data = torch.randn(10, 3, 128, 128)
        >>> target_data = torch.randn(10, 128, 128)
        >>> inp_patches, tgt_patches = quilt.unstitch_data_pair(input_data, target_data)
        >>> print(inp_patches.shape)  # (M, 3, 32, 32)
        >>> print(tgt_patches.shape)  # (M, 32, 32)
        >>> # With positional channels:
        >>> inp_patches, tgt_patches = quilt.unstitch_data_pair(
        ...     input_data, target_data, add_positional_channels=True
        ... )
        >>> print(inp_patches.shape)  # (M, 5, 32, 32) - 3 original + 2 positional channels
        """
        modsel = None
        if missing_label is not None:
            modsel = self.border_tensor() < 0.5

        rearranged = False
        if len(tensor_out.shape) == 3:
            tensor_out = einops.rearrange(tensor_out, "N Y X -> N () Y X")
            rearranged = True
        assert len(tensor_out.shape) == 4
        assert len(tensor_in.shape) == 4
        assert tensor_in.shape[0] == tensor_out.shape[0]

        if return_positions:
            unstitched_in, positions = self.unstitch(
                tensor_in,
                return_positions=True,
                include_n_position=include_n_position,
                add_positional_channels=add_positional_channels,
                normalize_positions=normalize_positions,
                position_mode=position_mode,
            )
            unstitched_out = self.unstitch(tensor_out)
        else:
            unstitched_in = self.unstitch(
                tensor_in,
                add_positional_channels=add_positional_channels,
                normalize_positions=normalize_positions,
                position_mode=position_mode,
            )
            unstitched_out = self.unstitch(tensor_out)

        if modsel is not None:
            unstitched_out[:, :, modsel] = missing_label

        if rearranged:
            assert unstitched_out.shape[1] == 1
            unstitched_out = unstitched_out.squeeze(dim=1)

        if return_positions:
            return unstitched_in, unstitched_out, positions
        return unstitched_in, unstitched_out

    def unstitch(
        self,
        tensor: torch.Tensor,
        return_positions: bool = False,
        include_n_position: bool = False,
        add_positional_channels: bool = False,
        normalize_positions: bool = True,
        position_mode: str = "absolute",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Split a tensor into smaller overlapping patches.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape (N, C, Y, X) where:
            - N: Number of images
            - C: Number of channels
            - Y: Height (must match self.Y)
            - X: Width (must match self.X)
        return_positions : bool, optional
            If True, also return positional embeddings. Default is False.
        include_n_position : bool, optional
            If True and return_positions=True, include N (batch) index in positions.
            If False, positions only contain [Y_pos, X_pos]. Default is False.
        add_positional_channels : bool, optional
            If True, add positional embedding channels directly to patches.
            Adds 2 channels (Y and X coordinates) to each patch.
            Default is False.
        normalize_positions : bool, optional
            If True and add_positional_channels=True, normalize coordinates to [0, 1].
            If False, use raw pixel coordinates. Default is True.
        position_mode : str, optional
            Mode for positional channels when add_positional_channels=True:
            - "absolute": Use absolute image coordinates (offset by patch position)
            - "relative": Use relative coordinates within patch (0 to window-1)
            Default is "absolute".

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            If return_positions=False and add_positional_channels=False:
            - Patches tensor of shape (M, C, window[0], window[1]) where:
              - M = N * nY * nX (total number of patches)
              - window[0], window[1]: Patch dimensions

            If return_positions=True:
            - patches: Tensor of shape (M, C, window[0], window[1]) or (M, C+2, window[0], window[1]) if add_positional_channels=True
            - positions: Tensor of shape (M, 2) or (M, 3) containing [Y_pos, X_pos] or [N_idx, Y_pos, X_pos]

            If add_positional_channels=True:
            - Patches tensor of shape (M, C+2, window[0], window[1]) with Y and X coordinate channels added

        Examples
        --------
        >>> quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16))
        >>> data = torch.randn(10, 3, 128, 128)
        >>> patches = quilt.unstitch(data)
        >>> print(patches.shape)  # (M, 3, 32, 32)
        >>> # With positional channels added:
        >>> patches = quilt.unstitch(data, add_positional_channels=True)
        >>> print(patches.shape)  # (M, 5, 32, 32) - 3 original + 2 positional channels
        >>> # With positional embeddings (Y, X only):
        >>> patches, positions = quilt.unstitch(data, return_positions=True)
        >>> print(positions.shape)  # (M, 2) - [Y_pos, X_pos]
        >>> # With N position included:
        >>> patches, positions = quilt.unstitch(data, return_positions=True, include_n_position=True)
        >>> print(positions.shape)  # (M, 3) - [N_idx, Y_pos, X_pos]
        >>> # Both positional channels and return positions:
        >>> patches, positions = quilt.unstitch(
        ...     data,
        ...     return_positions=True,
        ...     add_positional_channels=True,
        ...     position_mode="relative"
        ... )
        >>> print(patches.shape)  # (M, 5, 32, 32) - with relative positional channels
        """
        N, _C, _Y, _X = tensor.shape
        result = []
        positions_list = [] if return_positions else None

        for n in range(N):
            tmp = tensor[n, ...]
            for yy in range(self.nY):
                for xx in range(self.nX):
                    start_y = min(yy * self.step[0], self.Y - self.window[0])
                    start_x = min(xx * self.step[1], self.X - self.window[1])
                    stop_y = start_y + self.window[0]
                    stop_x = start_x + self.window[1]
                    patch = tmp[:, start_y:stop_y, start_x:stop_x]
                    result.append(patch)
                    if return_positions:
                        if include_n_position:
                            positions_list.append([n, start_y, start_x])
                        else:
                            positions_list.append([start_y, start_x])

        patches = einops.rearrange(result, "M C Y X -> M C Y X")

        # Add positional embedding channels if requested
        if add_positional_channels:
            M, C, U, V = patches.shape
            device = patches.device
            dtype = patches.dtype

            # Create coordinate grids
            y_coords = torch.arange(U, device=device, dtype=dtype)
            x_coords = torch.arange(V, device=device, dtype=dtype)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

            # Expand to batch dimension: (U, V) -> (M, 1, U, V)
            y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(M, 1, U, V)
            x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(M, 1, U, V)

            if position_mode == "absolute":
                # Add absolute positions (offset by patch position)
                for i in range(M):
                    if return_positions:
                        if include_n_position:
                            y_pos = positions_list[i][1]
                            x_pos = positions_list[i][2]
                        else:
                            y_pos = positions_list[i][0]
                            x_pos = positions_list[i][1]
                    else:
                        # Recompute position from patch index
                        patch_idx = i
                        patch_in_image = patch_idx % (self.nY * self.nX)
                        yy = patch_in_image // self.nX
                        xx = patch_in_image % self.nX
                        y_pos = min(yy * self.step[0], self.Y - self.window[0])
                        x_pos = min(xx * self.step[1], self.X - self.window[1])

                    y_grid[i, 0] += y_pos
                    x_grid[i, 0] += x_pos

                # Normalize if requested
                if normalize_positions:
                    y_grid = y_grid.float() / self.Y
                    x_grid = x_grid.float() / self.X
            else:  # relative mode
                # Use relative coordinates within patch (0 to U-1, 0 to V-1)
                if normalize_positions:
                    y_grid = y_grid.float() / (U - 1) if U > 1 else y_grid.float()
                    x_grid = x_grid.float() / (V - 1) if V > 1 else x_grid.float()

            # Concatenate positional channels: (M, C, U, V) + (M, 1, U, V) + (M, 1, U, V)
            patches = torch.cat([patches, y_grid, x_grid], dim=1)

        if return_positions:
            positions = torch.tensor(
                positions_list, dtype=torch.int64, device=tensor.device
            )
            return patches, positions
        return patches

    def stitch(
        self,
        ml_tensor: torch.Tensor,
        use_numba: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reassemble patches back into full-size tensors.

        This method takes patches produced by `unstitch()` and stitches them back
        together, averaging overlapping regions using a weight matrix. Border regions
        are downweighted according to `border_weight`.

        Typical workflow:

        1. Unstitch the data::

           patches = quilt.unstitch(input_images)

        2. Process patches with your model::

           output_patches = model(patches)

        3. Stitch back together::

           reconstructed, weights = quilt.stitch(output_patches)

        Parameters
        ----------
        ml_tensor : torch.Tensor
            Patches tensor of shape (M, C, window[0], window[1]) where:
            - M must equal N * nY * nX (number of patches)
            - C: Number of channels
            - window: Patch dimensions
        use_numba : bool, optional
            Whether to use Numba JIT compilation for faster stitching.
            Default is True (recommended for performance).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple of (reconstructed, weights) where:
            - reconstructed: Shape (N, C, Y, X) - the stitched result
            - weights: Shape (Y, X) - normalization weights (number of contributors per pixel)

        Notes
        -----
        **Important**: When working with classification outputs:

        - Apply softmax AFTER stitching, not before
        - Averaging softmaxed tensors ≠ softmax of averaged tensors
        - Process logits, stitch them, then apply softmax to the final result

        Example::

            # CORRECT:
            logits = model(patches)
            stitched_logits, _ = quilt.stitch(logits)
            probabilities = F.softmax(stitched_logits, dim=1)

            # WRONG:
            probs = F.softmax(model(patches), dim=1)
            result, _ = quilt.stitch(probs)  # This is incorrect!

        Examples
        --------
        >>> quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16))
        >>> data = torch.randn(10, 3, 128, 128)
        >>> patches = quilt.unstitch(data)
        >>> processed = model(patches)
        >>> reconstructed, weights = quilt.stitch(processed)
        >>> print(reconstructed.shape)  # (10, C, 128, 128)
        """
        N, C, _Y, _X = ml_tensor.shape
        # we now need to figure out how to stitch this back into what dimension
        times = self.nY * self.nX
        M_images = N // times
        assert N % times == 0
        if use_numba:
            return stitch_parallel_colored(
                ml_tensor,
                self.weight,
                self.window,
                self.step,
                self.Y,
                self.X,
                self.nY,
                self.nX,
            )

        result = torch.zeros((M_images, C, self.Y, self.X))
        norma = torch.zeros((self.Y, self.X))

        for m in range(M_images):
            for yy in range(self.nY):
                for xx in range(self.nX):
                    here_and_now = times * m + yy * self.nX + xx
                    start_y = min(yy * self.step[0], self.Y - self.window[0])
                    start_x = min(xx * self.step[1], self.X - self.window[1])
                    stop_y = start_y + self.window[0]
                    stop_x = start_x + self.window[1]
                    tmp = ml_tensor[here_and_now, ...]
                    result[m, :, start_y:stop_y, start_x:stop_x] += tmp * self.weight
                    if m == 0:
                        norma[start_y:stop_y, start_x:stop_x] += self.weight

        result = result / norma
        return result, norma
