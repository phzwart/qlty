from typing import List, Optional, Tuple, Union

import einops
import numpy as np
import numpy.typing as npt
import torch
import zarr

from qlty.base import (
    compute_border_tensor_numpy,
    compute_chunk_times,
    compute_weight_matrix_numpy,
    normalize_border,
    validate_border_weight,
)


class LargeNCYXQuilt:
    """
    This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
    This class is aimed at handling tensors of type (N, C, Y, X).

    This object is geared towards handling large datasets.
    """

    def __init__(
        self,
        filename: str,
        N: int,
        Y: int,
        X: int,
        window: Tuple[int, int],
        step: Tuple[int, int],
        border: Optional[Union[int, Tuple[int, int]]],
        border_weight: float = 0.1,
    ) -> None:
        """
        This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
        This class is aimed at handling tensors of type (N, C, Y, X).

        Parameters
        ----------
        filename: the base filename for storage.
        Y : number of elements in the Y direction
        X : number of elements in the X direction
        window: The size of the sliding window, a tuple (Ysub, Xsub)
        step: The step size at which we want to sample the sliding window (Ystep, Xstep)
        border: Border pixels of the window we want to 'ignore' or down weight when stitching things back
        border_weight: The weight for the border pixels, should be between 0 and 1. The default of 0.1 should be fine
        """
        self.filename = filename
        self.N = N
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

        # Compute weight matrix (as torch tensor for compatibility)
        weight_np = compute_weight_matrix_numpy(
            window=window, border=self.border, border_weight=self.border_weight
        )
        self.weight = torch.from_numpy(weight_np)

        self.N_chunks = self.N * self.nY * self.nX
        self.mean = None
        self.norma = None

        self.chunkerator = iter(np.arange(self.N_chunks))

    def border_tensor(self) -> npt.NDArray[np.float64]:
        """Compute border tensor indicating valid (non-border) regions."""
        return compute_border_tensor_numpy(window=self.window, border=self.border)

    def get_times(self) -> Tuple[int, int]:
        """
        Computes the number of chunks along Y and X dimensions, ensuring the last chunk
        is included by adjusting the starting points.
        """
        return compute_chunk_times(
            dimension_sizes=(self.Y, self.X), window=self.window, step=self.step
        )

    def unstitch_and_clean_sparse_data_pair(
        self,
        tensor_in: torch.Tensor,
        tensor_out: torch.Tensor,
        missing_label: Union[int, float],
    ) -> Tuple[Union[torch.Tensor, List], Union[torch.Tensor, List]]:
        """
        Split input and output tensors into patches, filtering out patches with no valid data.

        This method combines unstitching with sparse data filtering. It:
        1. Splits both tensors into patches
        2. Marks border regions as missing
        3. Filters out patches that contain only missing labels

        Parameters
        ----------
        tensor_in : torch.Tensor
            Input tensor of shape (N, C, Y, X). The tensor going into the network.
        tensor_out : torch.Tensor
            Output tensor of shape (N, C, Y, X) or (N, Y, X). The target tensor.
            Missing/invalid data should be marked with `missing_label`.
        missing_label : Union[int, float]
            Label value that indicates missing/invalid data. Patches containing only
            this value (including border regions) will be filtered out.

        Returns
        -------
        Tuple[Union[torch.Tensor, List], Union[torch.Tensor, List]]
            A tuple of (input_patches, output_patches). If no valid patches are found,
            returns empty lists. Otherwise returns torch.Tensor objects.

            - input_patches: Shape (M, C, window[0], window[1]) where M <= N * nY * nX
            - output_patches: Shape (M, C, window[0], window[1]) or (M, window[0], window[1])

        Notes
        -----
        - Border regions are automatically marked as missing in the output patches
        - Only patches with at least one non-missing label in the valid (non-border) region are kept
        - This is useful for training with sparse annotations where most of the image is unlabeled

        Examples
        --------
        >>> quilt = LargeNCYXQuilt("data", N=10, Y=128, X=128,
        ...                        window=(32, 32), step=(16, 16), border=(5, 5))
        >>> input_data = torch.randn(10, 3, 128, 128)
        >>> labels = torch.ones(10, 128, 128) * (-1)  # All missing
        >>> labels[:, 20:108, 20:108] = 1.0            # Some valid data
        >>> inp_patches, lbl_patches = quilt.unstitch_and_clean_sparse_data_pair(
        ...     input_data, labels, missing_label=-1
        ... )
        >>> print(f"Valid patches: {len(inp_patches) if isinstance(inp_patches, list) else inp_patches.shape[0]}")
        """
        rearranged = False

        if len(tensor_out.shape) == 3:
            tensor_out = tensor_out.unsqueeze(dim=1)
            rearranged = True
        assert len(tensor_out.shape) == 4
        assert len(tensor_in.shape) == 4
        assert tensor_in.shape[0] == tensor_out.shape[0]

        unstitched_in = []
        unstitched_out = []
        modsel = self.border_tensor()
        modsel = modsel < 0.5

        for ii in range(self.N_chunks):
            out_chunk = self.unstitch(tensor_out, ii).clone()
            out_chunk[:, modsel] = missing_label
            NN = out_chunk.nelement()
            not_present = torch.sum(out_chunk == missing_label).item()
            if not_present != NN:
                unstitched_in.append(self.unstitch(tensor_in, ii))
                unstitched_out.append(out_chunk)
        if len(unstitched_in) > 0:
            unstitched_in = einops.rearrange(unstitched_in, "N C Y X -> N C Y X")
            unstitched_out = einops.rearrange(unstitched_out, "N C Y X -> N C Y X")
            if rearranged:
                assert unstitched_out.shape[1] == 1
                unstitched_out = unstitched_out.squeeze(dim=1)
            return unstitched_in, unstitched_out
        else:
            return [], []

    def unstitch(self, tensor: torch.Tensor, index: int) -> torch.Tensor:
        """
        Extract a single patch from a tensor by index.

        This method is used internally by `unstitch_next()` but can also be called
        directly if you know the patch index.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape (N, C, Y, X) where:
            - N: Number of images
            - C: Number of channels
            - Y, X: Must match self.Y and self.X
        index : int
            Linear index of the patch to extract. Must be in range [0, N_chunks).

        Returns
        -------
        torch.Tensor
            Single patch of shape (C, window[0], window[1])

        Examples
        --------
        >>> quilt = LargeNCYXQuilt("data", N=10, Y=128, X=128,
        ...                        window=(32, 32), step=(16, 16))
        >>> data = torch.randn(10, 3, 128, 128)
        >>> patch = quilt.unstitch(data, index=0)
        >>> print(patch.shape)  # (3, 32, 32)
        """
        N, C, Y, X = tensor.shape

        out_shape = (N, self.nY, self.nX)
        n, yy, xx = np.unravel_index(index, out_shape)

        # Adjust the starting point for the last chunk in each dimension
        start_y = min(yy * self.step[0], Y - self.window[0])
        start_x = min(xx * self.step[1], X - self.window[1])

        stop_y = start_y + self.window[0]
        stop_x = start_x + self.window[1]

        patch = tensor[n, :, start_y:stop_y, start_x:stop_x]
        return patch

    def stitch(
        self,
        patch: torch.Tensor,
        index_flat: int,
        patch_var: Optional[torch.Tensor] = None,
    ) -> None:
        C = patch.shape[1]
        if self.mean is None:
            # Initialization code remains the same...
            self.mean = zarr.open(
                self.filename + "_mean_cache.zarr",
                shape=(self.N, C, self.Y, self.X),
                chunks=(1, C, self.window[0], self.window[1]),
                mode="w",
                fill_value=0,
            )

            self.std = zarr.open(
                self.filename + "_std_cache.zarr",
                shape=(self.N, C, self.Y, self.X),
                chunks=(1, C, self.window[0], self.window[1]),
                mode="w",
                fill_value=0,
            )

            self.norma = zarr.open(
                self.filename + "_norma_cache.zarr",
                shape=(self.Y, self.X),
                chunks=self.window,
                mode="w",
                fill_value=0,
            )

        screen_shape = (self.N, self.nY, self.nX)
        n, yy, xx = np.unravel_index(index_flat, screen_shape)
        # Adjust the starting point for the last chunk in each dimension
        start_y = min(yy * self.step[0], self.Y - self.window[0])
        start_x = min(xx * self.step[1], self.X - self.window[1])

        stop_y = start_y + self.window[0]
        stop_x = start_x + self.window[1]

        # Update the mean, std, and norma arrays
        self.mean[n : n + 1, :, start_y:stop_y, start_x:stop_x] += (
            patch.numpy() * self.weight.numpy()
        )
        if patch_var is not None:
            self.std[n : n + 1, :, start_y:stop_y, start_x:stop_x] += (
                patch_var.numpy() * self.weight.numpy()
            )

        if n == 0:
            self.norma[start_y:stop_y, start_x:stop_x] += self.weight.numpy()

    def unstitch_next(self, tensor: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Get the next patch in sequence (generator-like interface).

        This method maintains an internal iterator and returns the next patch
        each time it's called. Useful for processing large datasets chunk by chunk.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape (N, C, Y, X) where N matches self.N

        Returns
        -------
        Tuple[int, torch.Tensor]
            A tuple of (index, patch) where:
            - index: Linear index of the patch (0 to N_chunks-1)
            - patch: Patch tensor of shape (C, window[0], window[1])

        Notes
        -----
        The iterator resets after reaching the end. To process all patches::

            for i in range(quilt.N_chunks):
                index, patch = quilt.unstitch_next(data)
                # Process patch...

        Examples
        --------
        >>> quilt = LargeNCYXQuilt("data", N=10, Y=128, X=128,
        ...                        window=(32, 32), step=(16, 16))
        >>> data = torch.randn(10, 3, 128, 128)
        >>> for i in range(quilt.N_chunks):
        ...     idx, patch = quilt.unstitch_next(data)
        ...     processed = model(patch.unsqueeze(0))
        ...     quilt.stitch(processed, idx)
        """
        this_ind = next(self.chunkerator)
        tmp = self.unstitch(tensor, this_ind)
        return this_ind, tmp

    def return_mean(
        self, std: bool = False, normalize: bool = False, eps: float = 1e-8
    ) -> Union[
        npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ]:
        """
        Compute and return the final stitched result.

        After calling `stitch()` for all patches, this method computes the final
        averaged result. The result is normalized by the weight matrix to account
        for overlapping regions and border downweighting.

        Parameters
        ----------
        std : bool, optional
            Whether to compute and return the standard deviation. Requires that
            `patch_var` was provided to `stitch()` calls. Default is False.
        normalize : bool, optional
            Whether to normalize the result so that values sum to 1.0 along the
            channel dimension. Useful for probability distributions. Default is False.
        eps : float, optional
            Small epsilon value to prevent division by zero. Default is 1e-8.

        Returns
        -------
        Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]
            If std=False: Returns mean array of shape (N, C, Y, X)
            If std=True: Returns tuple (mean, std) where both have shape (N, C, Y, X)

            The result is a NumPy array (stored as Zarr array on disk).

        Notes
        -----
        - This method uses Dask for parallel processing of the Zarr arrays
        - Results are saved to disk as Zarr arrays (filename + '_mean.zarr' and '_std.zarr')
        - The computation happens lazily and is only executed when needed

        Examples
        --------
        >>> quilt = LargeNCYXQuilt("data", N=10, Y=128, X=128,
        ...                        window=(32, 32), step=(16, 16))
        >>> # ... process all patches with quilt.stitch() ...
        >>> mean = quilt.return_mean()
        >>> mean, std = quilt.return_mean(std=True)
        >>> print(f"Mean shape: {mean.shape}")  # (10, C, 128, 128)
        """
        import dask.array as da

        # Convert Zarr arrays to Dask arrays for parallel processing
        mean_dask = da.from_zarr(self.mean)
        norma_dask = da.from_zarr(self.norma) + eps
        norma_dask = da.expand_dims(norma_dask, axis=0)
        norma_dask = da.expand_dims(norma_dask, axis=0)
        std_dask = da.from_zarr(self.std) if std else None

        # Compute mean and std using Dask
        mean_accumulated = mean_dask / norma_dask
        if std:
            std_accumulated = da.sqrt(da.abs(std_dask / norma_dask))

        # Renormalize if required
        if normalize:
            norm = da.sum(mean_accumulated, axis=0)
            mean_accumulated /= norm
            if std:
                std_accumulated /= norm

        # Define file paths for Zarr arrays
        mean_zarr_path = self.filename + "_mean.zarr"
        std_zarr_path = (self.filename + "_std.zarr") if std else None

        # Store the result into Zarr arrays on disk
        mean_zarr = mean_accumulated.compute()
        zarr.save(mean_zarr_path, mean_zarr)
        if std:
            std_zarr = std_accumulated.compute()
            zarr.save(std_zarr_path, std_zarr)
            return mean_zarr, std_zarr
        return mean_zarr


def tst():
    data = np.random.uniform(0, 1, (2, 1, 300, 300)) * 100.0
    labels = np.zeros((2, 300, 300)) - 1
    labels[:, 0:151, 0:151] = 1
    Tdata = torch.Tensor(data)
    Tlabels = torch.Tensor(labels)

    qobj = LargeNCYXQuilt(
        "test200D",
        2,
        300,
        300,
        window=(50, 50),
        step=(25, 25),
        border=(10, 10),
        border_weight=1.0,
    )

    d, n = qobj.unstitch_and_clean_sparse_data_pair(Tdata, Tlabels, -1)

    assert d.shape[0] == 36 * 2
    for ii in range(qobj.N_chunks):
        ind, tmp = qobj.unstitch_next(Tdata)
        neural_network_result = tmp.unsqueeze(0)
        qobj.stitch(neural_network_result, ii)
    mean = qobj.return_mean()
    assert np.max(np.abs(mean - data)) < 1e-4

    qobj = LargeNCYXQuilt(
        "test200D",
        2,
        300,
        300,
        window=(150, 150),
        step=(25, 25),
        border=(10, 10),
        border_weight=1.0,
    )

    labels = np.zeros((2, 300, 300)) - 1
    labels[:, 0:51, 0:51] = 1
    Tlabels = torch.tensor(labels)
    d, n = qobj.unstitch_and_clean_sparse_data_pair(Tdata, Tlabels, -1)
    assert d.shape[0] == 8

    return True


if __name__ == "__main__":
    tst()
    print("OK")
