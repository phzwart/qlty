import torch
import zarr
import numpy as np
import einops


class LargeNCZYXQuilt(object):
    """
    This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
    This class is aimed at handling tensors of type (N,C,Z,Y,X)

    This object is geared towards handling large datasets.
    """

    def __init__(self,
                 filename,
                 N, Z, Y, X,
                 window,
                 step,
                 border,
                 border_weight=0.1,
                 ):
        """
        This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
        This class is aimed at handling tensors of type (N,C,Z,Y,X).

        Parameters
        ----------
        filename: the base filename for storage.
        Z : number of elements in the Z direction
        Y : number of elements in the Y direction
        X : number of elements in the X direction
        window: The size of the sliding window, a tuple (Zsub, Ysub, Xsub)
        step: The step size at which we want to sample the sliding window (Zstep, Ystep,Xstep)
        border: Border pixels of the window we want to 'ignore' or down weight when stitching things back
        border_weight: The weight for the border pixels, should be between 0 and 1. The default of 0.1 should be fine
        """
        border_weight = max(border_weight, 1e-8)
        self.filename = filename
        self.N = N
        self.Z = Z
        self.Y = Y
        self.X = X
        self.window = window
        self.step = step

        self.border = border
        self.border_weight = border_weight
        if border == 0:
            self.border = None
        assert self.border_weight <= 1.0
        assert self.border_weight >= 0.0

        self.nZ, self.nY, self.nX = self.get_times()

        self.weight = torch.ones(self.window)
        if self.border is not None:
            self.weight = torch.zeros(self.window) + border_weight
            self.weight[border[0]:-(border[0]),
                        border[1]:-(border[1]),
                        border[2]:-(border[2])
                        ] = 1.0

        self.N_chunks = self.N * self.nZ * self.nY * self.nX
        self.mean = None
        self.norma = None

        self.chunkerator = iter(np.arange(self.N_chunks))

    def border_tensor(self):
        result = np.zeros(self.window)
        result[self.border[0]:-(self.border[0]),
        self.border[1]:-(self.border[1]),
        self.border[2]:-(self.border[2])] = 1.0
        return result

    def get_times(self):
        """
        Computes how many steps along Z, Y and X we will take.

        Returns
        -------
        Z_step, Y_step, X_step: steps along the Z, Y and X direction
        """

        Z_times = (self.Z - self.window[0]) // self.step[0] + 1
        Y_times = (self.Y - self.window[1]) // self.step[1] + 1
        X_times = (self.X - self.window[2]) // self.step[2] + 1
        return Z_times, Y_times, X_times

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

        if len(tensor_out.shape) == 4:
            tensor_out = tensor_out.unsqueeze(dim=1)
            rearranged = True
        assert len(tensor_out.shape) == 5
        assert len(tensor_in.shape) == 5
        assert tensor_in.shape[0] == tensor_out.shape[0]

        unstitched_in = []
        unstitched_out = []
        for ii in range(self.N_chunks):
            out_chunk = self.unstitch(tensor_out, ii)
            tmp_out_chunk = out_chunk[:,
                            self.border[0]:-(self.border[0]),
                            self.border[1]:-(self.border[1]),
                            self.border[2]:-(self.border[2])]
            NN = tmp_out_chunk.nelement()
            not_present = torch.sum(tmp_out_chunk == missing_label).item()
            if not_present != NN:
                unstitched_in.append(self.unstitch(tensor_in, ii))
                unstitched_out.append(out_chunk)
        unstitched_in = einops.rearrange(unstitched_in, "N C Z Y X -> N C Z Y X")
        unstitched_out = einops.rearrange(unstitched_out, "N C Z Y X -> N C Z Y X")
        if rearranged:
            assert unstitched_out.shape[1] == 1
            unstitched_out = unstitched_out.squeeze(dim=1)
        return unstitched_in, unstitched_out

    def unstitch(self, tensor, index):
        """
        Unstich a single tensor.

        Parameters
        ----------
        tensor: input tensor to be chopped up
        index: the index of the chunk

        Returns
        -------
        A patched tensor
        """
        N, C, Z, Y, X = tensor.shape

        # figure out the right output size
        out_shape = (N, self.nZ, self.nY, self.nX)
        n, zz, yy, xx = np.unravel_index(index, out_shape)

        start_z = zz * self.step[0]
        start_y = yy * self.step[1]
        start_x = xx * self.step[2]

        stop_z = start_z + self.window[0]
        stop_y = start_y + self.window[1]
        stop_x = start_x + self.window[2]

        patch = tensor[n, :, start_z:stop_z, start_y:stop_y, start_x:stop_x]

        return patch

    def stitch(self, patch, index_flat, patch_var=None):
        """
        Stitch overlapping chunks back together.

        Parameters
        ----------
        patch : The chunk of data
        index_flat : the 'index' of the patch. Use a unrolled / flattened index.
                     We use np.unravel_index() to find its location

        Returns
        -------
        No return value. Once done iterating, use .return_mean() to get the average.


        """
        # build the zarr arrays where we need to store things if they are not there yet
        C = patch.shape[1]

        if self.mean is None:
            self.mean = zarr.open(self.filename + "_mean.zarr",
                                  shape=(self.N, C, self.Z, self.Y, self.X),
                                  chunks=(1, C, self.window[0], self.window[1], self.window[2]),
                                  mode='w', fill_value=0, )

            self.std = zarr.open(self.filename + "_std.zarr",
                                  shape=(self.N, C, self.Z, self.Y, self.X),
                                  chunks=(1, C, self.window[0], self.window[1], self.window[2]),
                                  mode='w', fill_value=0, )

            self.norma = zarr.open(self.filename + "_norma.zarr",
                                   shape=(self.Z, self.Y, self.X),
                                   chunks=self.window,
                                   mode='w', fill_value=0)

        screen_shape = (self.N, self.nZ, self.nY, self.nX)
        n, zz, yy, xx = np.unravel_index(index_flat, screen_shape)

        start_z = zz * self.step[0]
        start_y = yy * self.step[1]
        start_x = xx * self.step[2]
        stop_z = start_z + self.window[0]
        stop_y = start_y + self.window[1]
        stop_x = start_x + self.window[2]

        self.mean[n:n+1, :, start_z:stop_z, start_y:stop_y, start_x:stop_x] += patch.numpy() * self.weight
        if patch_var is not None:
            self.std[n:n+1, :, start_z:stop_z, start_y:stop_y, start_x:stop_x] += patch_var.numpy() * self.weight

        if n == 0:
            self.norma[start_z:stop_z, start_y:stop_y, start_x:stop_x] += self.weight

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

    def return_mean(self, std=False, normalize=True):
        """
        Return the averaged result.

        Returns
        -------
        The spatially averaged mean.
        """
        m = self.mean[...] / self.norma[...]
        n = 1.0
        if normalize:
            n = np.sum(m,axis=1)
        m = m / n
        if std:
            s = self.std[...] / self.norma[...]
            s = s / n
            return m, np.sqrt(np.abs(s))
        return m




def tst():
    data = np.random.uniform(0, 1, (1, 300, 300, 300))
    labels = np.zeros((1, 300, 300, 300)) - 1
    labels[:, 0:151, 0:151, 0:51] = 1
    Tdata = torch.Tensor(data).unsqueeze(dim=0)
    Tlabels = torch.Tensor(labels)

    qobj = LargeNCZYXQuilt("test", 1, 300, 300, 300,
                           window=(50, 50, 50),
                           step=(50, 50, 50),
                           border=(10, 10, 10))

    d, n = qobj.unstitch_and_clean_sparse_data_pair(Tdata, Tlabels, -1)
    assert d.shape[0] == 9
    for ii in range(qobj.N_chunks):
        ind, tmp = qobj.unstich_next(Tdata)
        neural_network_result = tmp
        qobj.stitch(neural_network_result,ii)
    mean = qobj.return_mean()
    assert np.max(np.abs(mean - data)) < 1e-4
    return True


if __name__ == "__main__":
    tst()
    print("OK")
