from typing import Optional, Tuple, Union

import einops
import torch

from qlty.base import (
    compute_border_tensor_torch,
    compute_chunk_times,
    compute_weight_matrix_torch,
    normalize_border,
    validate_border_weight,
)


class NCZYXQuilt:
    """
    This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
    This class is aimed at handling tensors of type (N,C,Z,Y,X)

    """

    def __init__(
        self,
        Z: int,
        Y: int,
        X: int,
        window: Tuple[int, int, int],
        step: Tuple[int, int, int],
        border: Optional[Union[int, Tuple[int, int, int]]],
        border_weight: float = 0.1,
    ) -> None:
        """
        This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
        This class is aimed at handling tensors of type (N,C,Z,Y,X).

        Parameters
        ----------
        Z : number of elements in the Z direction
        Y : number of elements in the Y direction
        X : number of elements in the X direction
        window: The size of the sliding window, a tuple (Zsub, Ysub, Xsub)
        step: The step size at which we want to sample the sliding window (Zstep, Ystep,Xstep)
        border: Border pixels of the window we want to 'ignore' or down weight when stitching things back
        border_weight: The weight for the border pixels, should be between 0 and 1. The default of 0.1 should be fine
        """
        self.Z = Z
        self.Y = Y
        self.X = X
        self.window = window
        self.step = step

        # Normalize and validate border
        self.border = normalize_border(border, ndim=3)
        self.border_weight = validate_border_weight(border_weight)

        # Compute chunk times
        self.nZ, self.nY, self.nX = compute_chunk_times(
            dimension_sizes=(Z, Y, X), window=window, step=step
        )

        # Compute weight matrix
        self.weight = compute_weight_matrix_torch(
            window=window, border=self.border, border_weight=self.border_weight
        )

    def border_tensor(self) -> torch.Tensor:
        """Compute border tensor indicating valid (non-border) regions."""
        return compute_border_tensor_torch(window=self.window, border=self.border)

    def get_times(self) -> Tuple[int, int, int]:
        """
        Computes the number of chunks along Z, Y, and X dimensions, ensuring the last chunk
        is included by adjusting the starting points.
        """
        return compute_chunk_times(
            dimension_sizes=(self.Z, self.Y, self.X), window=self.window, step=self.step
        )

    def unstitch_data_pair(
        self, tensor_in: torch.Tensor, tensor_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split input and output 3D tensors into smaller overlapping patches.

        This method is useful for training neural networks on 3D volumes where you need
        to process input-output pairs together.

        Parameters
        ----------
        tensor_in : torch.Tensor
            Input tensor of shape (N, C, Z, Y, X). The tensor going into the network.
        tensor_out : torch.Tensor
            Output tensor of shape (N, C, Z, Y, X) or (N, Z, Y, X). The target tensor.
            If 4D, will be automatically expanded to 5D.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple of (input_patches, output_patches) where:
            - input_patches: Shape (M, C, window[0], window[1], window[2])
            - output_patches: Shape (M, C, window[0], window[1], window[2]) or (M, window[0], window[1], window[2])
            where M = N * nZ * nY * nX

        Examples
        --------
        >>> quilt = NCZYXQuilt(Z=64, Y=64, X=64, window=(32, 32, 32), step=(16, 16, 16))
        >>> input_data = torch.randn(5, 1, 64, 64, 64)
        >>> target_data = torch.randn(5, 64, 64, 64)
        >>> inp_patches, tgt_patches = quilt.unstitch_data_pair(input_data, target_data)
        >>> print(inp_patches.shape)  # (M, 1, 32, 32, 32)
        >>> print(tgt_patches.shape)  # (M, 32, 32, 32)
        """
        rearranged = False
        if len(tensor_out.shape) == 4:
            tensor_out = einops.rearrange(tensor_out, "N Z Y X -> N () Z Y X")
            rearranged = True
        assert len(tensor_out.shape) == 5
        assert len(tensor_in.shape) == 5
        assert tensor_in.shape[0] == tensor_out.shape[0]

        unstitched_in = self.unstitch(tensor_in)
        unstitched_out = self.unstitch(tensor_out)
        if rearranged:
            assert unstitched_out.shape[1] == 1
            unstitched_out = unstitched_out.squeeze(dim=1)
        return unstitched_in, unstitched_out

    def unstitch(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split a 3D tensor into smaller overlapping patches.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape (N, C, Z, Y, X) where:
            - N: Number of volumes
            - C: Number of channels
            - Z, Y, X: Dimensions (must match self.Z, self.Y, self.X)

        Returns
        -------
        torch.Tensor
            Patches tensor of shape (M, C, window[0], window[1], window[2]) where:
            - M = N * nZ * nY * nX (total number of patches)
            - window[0], window[1], window[2]: Patch dimensions in Z, Y, X

        Examples
        --------
        >>> quilt = NCZYXQuilt(Z=64, Y=64, X=64, window=(32, 32, 32), step=(16, 16, 16))
        >>> volume = torch.randn(5, 1, 64, 64, 64)
        >>> patches = quilt.unstitch(volume)
        >>> print(patches.shape)  # (M, 1, 32, 32, 32)
        """
        N, C, Z, Y, X = tensor.shape
        result = []
        for n in range(N):
            tmp = tensor[n, ...]
            for zz in range(self.nZ):
                for yy in range(self.nY):
                    for xx in range(self.nX):
                        start_z = zz * self.step[0]
                        start_y = yy * self.step[1]
                        start_x = xx * self.step[2]

                        stop_z = start_z + self.window[0]
                        stop_y = start_y + self.window[1]
                        stop_x = start_x + self.window[2]

                        patch = tmp[:, start_z:stop_z, start_y:stop_y, start_x:stop_x]
                        result.append(patch)
        result = einops.rearrange(result, "M C Z Y X -> M C Z Y X")
        return result

    def stitch(self, ml_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reassemble 3D patches back into full-size volumes.

        This method takes patches produced by `unstitch()` and stitches them back
        together, averaging overlapping regions using a weight matrix.

        Typical workflow:

        1. Unstitch the data::

           patches = quilt.unstitch(volumes)

        2. Process patches with your model::

           output_patches = model(patches)

        3. Stitch back together::

           reconstructed, weights = quilt.stitch(output_patches)

        Parameters
        ----------
        ml_tensor : torch.Tensor
            Patches tensor of shape (M, C, window[0], window[1], window[2]) where:
            - M must equal N * nZ * nY * nX (number of patches)
            - C: Number of channels
            - window: Patch dimensions in (Z, Y, X)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple of (reconstructed, weights) where:
            - reconstructed: Shape (N, C, Z, Y, X) - the stitched result
            - weights: Shape (Z, Y, X) - normalization weights

        Notes
        -----
        **Important**: When working with classification outputs:

        - Apply softmax AFTER stitching, not before
        - Averaging softmaxed tensors ≠ softmax of averaged tensors
        - Process logits, stitch them, then apply softmax to the final result

        Examples
        --------
        >>> quilt = NCZYXQuilt(Z=64, Y=64, X=64, window=(32, 32, 32), step=(16, 16, 16))
        >>> volume = torch.randn(5, 1, 64, 64, 64)
        >>> patches = quilt.unstitch(volume)
        >>> processed = model(patches)
        >>> reconstructed, weights = quilt.stitch(processed)
        >>> print(reconstructed.shape)  # (5, C, 64, 64, 64)
        """
        N, C, Z, Y, X = ml_tensor.shape
        # we now need to figure out how to sticth this back into what dimension
        times = self.nZ * self.nY * self.nX
        M_images = N // times
        assert N % times == 0
        result = torch.zeros((M_images, C, self.Z, self.Y, self.X))
        norma = torch.zeros((self.Z, self.Y, self.X))

        this_image = 0
        for m in range(M_images):
            count = 0
            for zz in range(self.nZ):
                for yy in range(self.nY):
                    for xx in range(self.nX):
                        here_and_now = times * this_image + count

                        start_z = zz * self.step[0]
                        start_y = yy * self.step[1]
                        start_x = xx * self.step[2]
                        stop_z = start_z + self.window[0]
                        stop_y = start_y + self.window[1]
                        stop_x = start_x + self.window[2]

                        tmp = ml_tensor[here_and_now, ...]
                        result[
                            this_image,
                            :,
                            start_z:stop_z,
                            start_y:stop_y,
                            start_x:stop_x,
                        ] += tmp * self.weight
                        count += 1
                        # get the weight matrix, only compute once
                        if m == 0:
                            norma[
                                start_z:stop_z, start_y:stop_y, start_x:stop_x
                            ] += self.weight

            this_image += 1
        result = result / norma
        return result, norma
