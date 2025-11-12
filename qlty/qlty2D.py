import math
from typing import Optional, Tuple, Union

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
    ml_tensor, result, norma, weight, window, step, Y, X, nX, times, m
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
    window: Tuple[int, int],
    step: Tuple[int, int],
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
        (M_images, ml_tensor_np.shape[1], Y, X), dtype=ml_tensor_np.dtype
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
    window: Tuple[int, int],
    step: Tuple[int, int],
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
        (M_images, ml_tensor_np.shape[1], Y, X), dtype=ml_tensor_np.dtype
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
        window: Tuple[int, int],
        step: Tuple[int, int],
        border: Optional[Union[int, Tuple[int, int]]],
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
            dimension_sizes=(Y, X), window=window, step=step
        )

        # Compute weight matrix
        self.weight = compute_weight_matrix_torch(
            window=window, border=self.border, border_weight=self.border_weight
        )

    def border_tensor(self) -> torch.Tensor:
        """Compute border tensor indicating valid (non-border) regions."""
        return compute_border_tensor_torch(window=self.window, border=self.border)

    def get_times(self) -> Tuple[int, int]:
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
            dimension_sizes=(self.Y, self.X), window=self.window, step=self.step
        )

    def unstitch_data_pair(
        self,
        tensor_in: torch.Tensor,
        tensor_out: torch.Tensor,
        missing_label: Optional[Union[int, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple of (input_patches, output_patches) where:
            - input_patches: Shape (M, C, window[0], window[1])
            - output_patches: Shape (M, C, window[0], window[1]) or (M, window[0], window[1])
            where M = N * nY * nX

        Examples
        --------
        >>> quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5))
        >>> input_data = torch.randn(10, 3, 128, 128)
        >>> target_data = torch.randn(10, 128, 128)
        >>> inp_patches, tgt_patches = quilt.unstitch_data_pair(input_data, target_data)
        >>> print(inp_patches.shape)  # (M, 3, 32, 32)
        >>> print(tgt_patches.shape)  # (M, 32, 32)
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

        unstitched_in = self.unstitch(tensor_in)
        unstitched_out = self.unstitch(tensor_out)
        if modsel is not None:
            unstitched_out[:, :, modsel] = missing_label

        if rearranged:
            assert unstitched_out.shape[1] == 1
            unstitched_out = unstitched_out.squeeze(dim=1)
        return unstitched_in, unstitched_out

    def unstitch(self, tensor: torch.Tensor) -> torch.Tensor:
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

        Returns
        -------
        torch.Tensor
            Patches tensor of shape (M, C, window[0], window[1]) where:
            - M = N * nY * nX (total number of patches)
            - window[0], window[1]: Patch dimensions

        Examples
        --------
        >>> quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16))
        >>> data = torch.randn(10, 3, 128, 128)
        >>> patches = quilt.unstitch(data)
        >>> print(patches.shape)  # (M, 3, 32, 32)
        """
        N, C, Y, X = tensor.shape
        result = []

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
        result = einops.rearrange(result, "M C Y X -> M C Y X")
        return result

    def stitch(
        self, ml_tensor: torch.Tensor, use_numba: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        N, C, Y, X = ml_tensor.shape
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
