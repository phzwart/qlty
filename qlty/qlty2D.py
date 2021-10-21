import torch
import einops

class NCYXQuilt(object):
    """
    This class allows one to split larger tensors into smaller ones that perhaps do fit into memory.
    This class is aimed at handling tensors of type (N,C,Y,X)

    """
    def __init__(self,Y,X,window,step,border, border_weight=0.1):
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
        self.nY, self.nX = self.get_times()

        self.weight = torch.zeros( self.window ) + border_weight
        self.weight[border[0]:-(border[0]-1), border[1]:-(border[1]-1)] = 1.0 - border_weight

    def get_times(self):
        """
        Computes how many stpes along Y and along X we will take.

        Returns
        -------
        Y_step, X_step: steps along the Y and X direction
        """
        Y_times = (self.Y - self.window[0])//self.step[0] + 1
        X_times = (self.X - self.window[1])//self.step[1] + 1
        return Y_times, X_times

    def unstitch_data_pair(self, tensor_in, tensor_out):
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
        rearranged=False
        if len(tensor_out.shape)==3:
            tensor_out = einops.rearrange(tensor_out, "N Y X -> N () Y X")
            rearranged = True
        assert len(tensor_out.shape) == 4
        assert len(tensor_in.shape) == 4
        assert tensor_in.shape[0] == tensor_out.shape[0]

        unstitched_in = self.unstitch(tensor_in)
        unstitched_out = self.unstitch(tensor_out)
        if rearranged:
            assert unstitched_out.shape[1]==1
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
        N,C,Y,X = tensor.shape
        result = []
        for n in range(N):
            tmp = tensor[n,...]
            for yy in range(self.nY):
                for xx in range(self.nX):
                    start_y = yy*self.step[0]
                    start_x = xx*self.step[1]
                    stop_y = start_y + self.window[0]
                    stop_x = start_x + self.window[1]
                    patch = tmp[:, start_y:stop_y, start_x:stop_x]
                    result.append(patch)
        result = einops.rearrange(result, "M C Y X -> M C Y X")
        return result

    def stitch(self,ml_tensor):
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
        # we now need to figure out how to sticth this back into what dimension
        times = self.nY*self.nX
        M_images = N//times
        assert N%times==0
        result = torch.zeros( (M_images, C, self.Y, self.X) )
        norma = torch.zeros( (self.Y, self.X) )
        count = 0
        this_image = 0
        for m in range(M_images):
            count = 0
            for yy in range(self.nY):
                for xx in range(self.nX):

                    here_and_now = times*this_image+count
                    start_y = yy*self.step[0]
                    start_x = xx*self.step[1]
                    stop_y = start_y + self.window[0]
                    stop_x = start_x + self.window[1]

                    tmp = ml_tensor[here_and_now,...]
                    result[this_image,:,start_y:stop_y, start_x:stop_x] += tmp*self.weight
                    count += 1
                    # get the weight matrix, only compute once
                    if m==0:
                        norma[start_y:stop_y, start_x:stop_x]+=self.weight

            this_image+=1
        result = result/(norma)
        return result, norma