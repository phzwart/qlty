import torch
from numba import njit, prange
import einops


@njit(fastmath=True, parallel=True)
def numba_njit_stitch(ml_tensor, result, norma, weight, window, step, Y, X, nX, times, m):
    
    for i in prange(times):
        yy = i // nX
        xx = i % nX
        here_and_now = times * m + yy * nX + xx
        start_y = min(yy * step[0], Y - window[0])
        start_x = min(xx * step[1], X - window[1])
        stop_y = start_y + window[0]
        stop_x = start_x + window[1]
        for j in prange(ml_tensor.shape[1]):
            tmp = ml_tensor[here_and_now, j, ...]
            result[m, j, start_y:stop_y, start_x:stop_x] += tmp * weight
        # get the weight matrix, only compute once
        if m == 0:
            norma[start_y:stop_y, start_x:stop_x] += weight
    return result, norma


class NCYXQuilt(object):
    """
    This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
    This class is aimed at handling tensors of type (N,C,Y,X)

    """

    def __init__(self, Y, X, window, step, border, border_weight=1.0):
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
        border_weight = max(border_weight, 1e-8)
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
            self.weight[border[0]:-(border[0]), border[1]:-(border[1])] = 1.0

    def border_tensor(self):
        if self.border is not None:
            result = torch.zeros(self.window)
            result[self.border[0]:-(self.border[0]), self.border[1]:-(self.border[1])] = 1.0
        else:
            result = torch.ones(self.window)
        return result

    def get_times(self):
        """
        Computes the number of chunks along Z, Y, and X dimensions, ensuring the last chunk
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

        Y_times = compute_steps(self.Y, self.window[-2], self.step[-2])
        X_times = compute_steps(self.X, self.window[-1], self.step[-1])
        return Y_times, X_times

    def unstitch_data_pair(self, tensor_in, tensor_out, missing_label=None):
        """
        Take a tensor and split it in smaller overlapping tensors.
        If you train a network, tensor_in is the input, while tensor_out is the target tensor.

        Parameters
        ----------
        tensor_in: The tensor going into the network
        tensor_out: The tensor we train against

        Returns
        -------
        Tensor patches.
        """
        modsel = None
        if missing_label is not None:
            modsel = self.border_tensor() < 0.5
            modsel = modsel
            print(modsel.shape)


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
            unstitched_out[:,:,modsel]=missing_label

        if rearranged:
            assert unstitched_out.shape[1] == 1
            unstitched_out = unstitched_out.squeeze(dim=1)
        return unstitched_in, unstitched_out

    def unstitch(self, tensor):
        """
        Unstich a single tensor.

        Parameters
        ----------
        tensor

        Returns
        -------
        A patched tensor
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

    def stitch(self, ml_tensor, use_numba=True):
        """
        The assumption here is that we have done the following:

        1. unstitch the data
        patched_input_images = qlty_object.unstitch(input_images)

        2. run the network you have trained
        output_predictions = my_network(patched_input_images)

        3. Restitch the images back together, while averaging the overlapping regions
        prediction = qlty_object.stitch(output_predictions)

        Be careful when you apply a softmax (or equivalent) btw, as averaging softmaxed tensors are not likely to be
        equal to softmaxed averaged tensors. Worthwhile playing to figure out what works best.

        Parameters
        ----------
        ml_tensor

        Returns
        -------

        """
        N, C, Y, X = ml_tensor.shape
        # we now need to figure out how to stitch this back into what dimension
        times = self.nY * self.nX
        M_images = N // times
        assert N % times == 0
        result = torch.zeros((M_images, C, self.Y, self.X))
        norma = torch.zeros((self.Y, self.X))
        # needed for numba implementation
        if use_numba:
            ml_tensor = ml_tensor.numpy()
            result = result.numpy()
            norma = norma.numpy()
            weight = self.weight.numpy()
            
        this_image = 0
        
        for m in range(M_images):
            
            # numba jit implementation
            if use_numba:
                result, norma = numba_njit_stitch(ml_tensor, result, norma, weight, self.window, self.step, self.Y, self.X, self.nX, times, m)           
        
            # original implementation (modified)
            if use_numba == False:
                for yy in range(self.nY):
                    for xx in range(self.nX):
                        here_and_now = times * m + yy * self.nX + xx
                        start_y = min(yy * self.step[0], self.Y - self.window[0])
                        start_x = min(xx * self.step[1], self.X - self.window[1])
                        stop_y = start_y + self.window[0]
                        stop_x = start_x + self.window[1]
                        tmp = ml_tensor[here_and_now, ...]
                        result[m, :, start_y:stop_y, start_x:stop_x] += tmp * self.weight
                        # get the weight matrix, only compute once
                        if m == 0:
                            norma[start_y:stop_y, start_x:stop_x] += self.weight
                            
        # with numba implementation
        if use_numba:
            result = torch.tensor(result)
            norma = torch.tensor(norma)
            
        result = result / norma
        return result, norma
