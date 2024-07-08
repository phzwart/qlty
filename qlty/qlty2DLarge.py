import torch
import zarr
import numpy as np
import einops
import dask.array as da

class LargeNCYXQuilt(object):
    """
    This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
    This class is aimed at handling tensors of type (N, C, Y, X).

    This object is geared towards handling large datasets.
    """

    def __init__(self,
                 filename,
                 N, Y, X,
                 window,
                 step,
                 border,
                 border_weight=0.1,
                 ):
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
        border_weight = max(border_weight, 1e-8)
        self.filename = filename
        self.N = N
        self.Y = Y
        self.X = X
        self.window = window
        self.step = step

        self.border = border
        self.border_weight = border_weight
        if border == 0 or border == (0, 0):
            self.border = None
        assert self.border_weight <= 1.0
        assert self.border_weight >= 0.0

        self.nY, self.nX = self.get_times()

        self.weight = torch.ones(self.window)
        if self.border is not None:
            self.weight = torch.zeros(self.window) + border_weight
            self.weight[border[0]:-(border[0]),
                        border[1]:-(border[1])
                        ] = 1.0

        self.N_chunks = self.N * self.nY * self.nX
        self.mean = None
        self.norma = None

        self.chunkerator = iter(np.arange(self.N_chunks))

    def border_tensor(self):
        result = np.zeros(self.window)
        result[self.border[0]:-(self.border[0]),
               self.border[1]:-(self.border[1])] = 1.0
        return result

    def get_times(self):
        """
        Computes the number of chunks along Y and X dimensions, ensuring the last chunk
        is included by adjusting the starting points.
        """

        def compute_steps(dimension_size, window_size, step_size):
            # Calculate the number of full steps
            full_steps = (dimension_size - window_size) // step_size
            # Check if there is enough space left for the last chunk
            if dimension_size > full_steps * step_size + window_size:
                return full_steps + 2
            else:
                return full_steps + 1

        Y_times = compute_steps(self.Y, self.window[0], self.step[0])
        X_times = compute_steps(self.X, self.window[1], self.step[1])
        return Y_times, X_times

    def unstitch_and_clean_sparse_data_pair(self, tensor_in, tensor_out, missing_label):
        """
        Take a tensor and split it in smaller overlapping tensors.
        If you train a network, tensor_in is the input, while tensor_out is the target tensor.

        Parameters
        ----------
        tensor_in: The tensor going into the network
        tensor_out: The tensor we train against
        missing_label: if tensor_out elements contains this value, it is considered as not observed.
                       If a complete chunk only contains missing_label, it will not be used for training.
                       If a label that isn't missing_label is in the border area, it is treated as missing.

        Returns
        -------
        Tensor patches.
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
        for ii in range(self.N_chunks):
            out_chunk = self.unstitch(tensor_out, ii)
            tmp_out_chunk = out_chunk[:,
                            self.border[0]:-(self.border[0]),
                            self.border[1]:-(self.border[1])]
            NN = tmp_out_chunk.nelement()
            not_present = torch.sum(tmp_out_chunk == missing_label).item()
            if not_present != NN:
                unstitched_in.append(self.unstitch(tensor_in, ii))
                unstitched_out.append(out_chunk)
        unstitched_in = einops.rearrange(unstitched_in, "N C Y X -> N C Y X")
        unstitched_out = einops.rearrange(unstitched_out, "N C Y X -> N C Y X")
        if rearranged:
            assert unstitched_out.shape[1] == 1
            unstitched_out = unstitched_out.squeeze(dim=1)
        return unstitched_in, unstitched_out

    def unstitch(self, tensor, index):
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

    def stitch(self, patch, index_flat, patch_var=None):
        C = patch.shape[1]
        if self.mean is None:
            # Initialization code remains the same...
            self.mean = zarr.open(self.filename + "_mean_cache.zarr",
                                  shape=(self.N, C, self.Y, self.X),
                                  chunks=(1, C, self.window[0], self.window[1]),
                                  mode='w', fill_value=0, )

            self.std = zarr.open(self.filename + "_std_cache.zarr",
                                 shape=(self.N, C, self.Y, self.X),
                                 chunks=(1, C, self.window[0], self.window[1]),
                                 mode='w', fill_value=0, )

            self.norma = zarr.open(self.filename + "_norma_cache.zarr",
                                   shape=(self.Y, self.X),
                                   chunks=self.window,
                                   mode='w', fill_value=0)

        screen_shape = (self.N, self.nY, self.nX)
        n, yy, xx = np.unravel_index(index_flat, screen_shape)
        # Adjust the starting point for the last chunk in each dimension
        start_y = min(yy * self.step[0], self.Y - self.window[0])
        start_x = min(xx * self.step[1], self.X - self.window[1])

        stop_y = start_y + self.window[0]
        stop_x = start_x + self.window[1]

        # Update the mean, std, and norma arrays
        self.mean[n:n+1, :, start_y:stop_y, start_x:stop_x] += patch.numpy() * self.weight.numpy()
        if patch_var is not None:
            self.std[n:n+1, :, start_y:stop_y, start_x:stop_x] += patch_var.numpy() * self.weight.numpy()

        if n == 0:
            self.norma[start_y:stop_y, start_x:stop_x] += self.weight.numpy()

    def unstich_next(self, tensor):
        """
        Find the next unstitched chunk.

        Parameters
        ----------
        tensor : Tensor with data

        Returns
        -------
        A tensor with data.
        """
        this_ind = next(self.chunkerator)
        tmp = self.unstitch(tensor, this_ind)
        return this_ind, tmp    

    def return_mean(self, std=False, normalize=False, eps=1e-8):
        """
        Return the result

        Parameters
        ----------
        std : bool, optional
            Whether to compute and return the standard deviation.
        normalize : bool, optional
            Whether to normalize the mean.

        Returns
        -------
        The spatially averaged mean stored as a Zarr array.
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
            std_accumulated = da.sqrt(da.abs(da.sum(std_dask / norma_dask, axis=0)))

        # Renormalize if required
        if normalize:
            norm = da.sum(mean_accumulated, axis=0)
            mean_accumulated /= norm
            if std:
                std_accumulated /= norm

        # Define file paths for Zarr arrays
        mean_zarr_path = (self.filename + '_mean.zarr')
        std_zarr_path = (self.filename + '_std.zarr') if std else None

        # Store the result into Zarr arrays on disk
        mean_zarr = mean_accumulated.compute()
        zarr.save(mean_zarr_path, mean_zarr)
        if std:
            std_zarr = std_accumulated.compute()
            zarr.save(std_zarr_path, std_zarr)
            return mean_zarr, std_zarr
        return mean_zarr


def tst():
    data = np.random.uniform(0, 1, (2, 1, 300, 300))*100.0
    labels = np.zeros((2, 300, 300)) - 1
    labels[:, 0:151, 0:151] = 1
    Tdata = torch.Tensor(data)
    Tlabels = torch.Tensor(labels)

    qobj = LargeNCYXQuilt("test200D", 2, 300, 300,
                          window=(50, 50),
                          step=(25, 25),
                          border=(10,10), border_weight=1.0)

    d, n = qobj.unstitch_and_clean_sparse_data_pair(Tdata, Tlabels, -1)
    assert d.shape[0] == 36*2
    for ii in range(qobj.N_chunks):
        ind, tmp = qobj.unstich_next(Tdata)
        neural_network_result = tmp.unsqueeze(0)
        qobj.stitch(neural_network_result, ii)
    mean = qobj.return_mean()
    assert np.max(np.abs(mean - data)) < 1e-4
    return True


if __name__ == "__main__":
    tst()
    print("OK")
