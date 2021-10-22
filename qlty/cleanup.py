import torch
import einops

def weed_sparse_classification_training_pairs_2D(tensor_in, tensor_out, missing_label, border_tensor):
    """
    After tensors have been unstitched, we want want to be able to remove patches that have no data.
    To this extent, we inspect every patch and remove any that do not contain any data. In additon, we remove
    observations that lie in the border area. For this to work, a border_tensor must be supplied.

    The selection is made on the basis of the supplied 'tensor_out' data field.

    Parameters
    ----------
    tensor_in: input tensor
    tensor_out: output tensor
    missing_label: missing label flag (typically -1)
    border_tensor: the border tensor, obtained from the NCXYQuilt or NCZYXQuilt class

    Returns
    -------
    A list of tensors that does have valid training data.
    """

    tmp = torch.clone(tensor_out)
    sel = tmp==missing_label
    sel = sel*border_tensor

    if len(border_tensor.shape)==2:
        sel = einops.reduce( sel, "N Y X -> N", reduction='sum')

    if len(border_tensor.shape)==3:
        sel = einops.reduce( sel, "N C Y X -> N", reduction='sum')
    sel = sel.type(torch.bool)
    newin = tensor_in[~sel,...]
    newout = tensor_out[~sel,...]
    return newin, newout

