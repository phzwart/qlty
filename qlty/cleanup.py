from typing import Tuple, Union

import einops
import torch


def weed_sparse_classification_training_pairs_2D(
    tensor_in: torch.Tensor,
    tensor_out: torch.Tensor,
    missing_label: Union[int, float],
    border_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter out patches that contain no valid data after unstitching.

    This function removes patches that have only missing labels (or only in border
    regions). Useful for training with sparse annotations where most of the image
    is unlabeled.

    Parameters
    ----------
    tensor_in : torch.Tensor
        Input patches tensor, typically of shape (N, C, Y, X) or (N, Y, X)
    tensor_out : torch.Tensor
        Output patches tensor with labels, shape (N, C, Y, X) or (N, Y, X).
        Missing/invalid data should be marked with `missing_label`.
    missing_label : Union[int, float]
        Label value that indicates missing/invalid data (typically -1)
    border_tensor : torch.Tensor
        Border mask tensor from `NCYXQuilt.border_tensor()` or `NCZYXQuilt.border_tensor()`.
        Shape should be (Y, X) for 2D or (Z, Y, X) for 3D (this function handles 2D).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple of (valid_input, valid_output, removal_mask) where:
        - valid_input: Filtered input patches (only patches with valid data)
        - valid_output: Filtered output patches (only patches with valid data)
        - removal_mask: Boolean tensor indicating which patches were removed

    Notes
    -----
    - Only patches with at least one non-missing label in the valid (non-border) region are kept
    - Border regions are automatically excluded from the validity check
    - Useful for semi-supervised learning with sparse annotations

    Examples
    --------
    >>> from qlty import NCYXQuilt, weed_sparse_classification_training_pairs_2D
    >>> quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5))
    >>> input_patches = torch.randn(100, 3, 32, 32)
    >>> label_patches = torch.ones(100, 32, 32) * (-1)  # All missing
    >>> label_patches[0:50] = 1.0  # Some valid
    >>> border_tensor = quilt.border_tensor()
    >>> valid_in, valid_out, mask = weed_sparse_classification_training_pairs_2D(
    ...     input_patches, label_patches, missing_label=-1, border_tensor=border_tensor
    ... )
    >>> print(f"Kept {valid_in.shape[0]} out of {input_patches.shape[0]} patches")
    """

    tmp = torch.clone(tensor_out)
    sel = (tmp != missing_label).type(torch.int)

    # Expand border_tensor to match tensor_out shape if needed
    if len(border_tensor.shape) == 2 and len(tensor_out.shape) == 4:
        # tensor_out has channels, expand border_tensor
        border_tensor = border_tensor.unsqueeze(0).unsqueeze(0)
        sel = sel * border_tensor
        sel = einops.reduce(sel, "N C Y X -> N", reduction="sum")
    elif len(border_tensor.shape) == 2:
        # tensor_out is (N, Y, X)
        border_tensor = border_tensor.unsqueeze(0)
        sel = sel * border_tensor
        sel = einops.reduce(sel, "N Y X -> N", reduction="sum")
    elif len(border_tensor.shape) == 3:
        # tensor_out is (N, C, Y, X)
        border_tensor = border_tensor.unsqueeze(0)
        sel = sel * border_tensor
        sel = einops.reduce(sel, "N C Y X -> N", reduction="sum")
    else:
        # Fallback: multiply and reduce
        sel = sel * border_tensor
        sel = sel.sum(dim=tuple(range(1, len(sel.shape))))

    sel = sel == 0
    newin = tensor_in[~sel, ...]
    newout = tensor_out[~sel, ...]
    return newin, newout, sel


def weed_sparse_classification_training_pairs_3D(
    tensor_in: torch.Tensor,
    tensor_out: torch.Tensor,
    missing_label: Union[int, float],
    border_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter out 3D patches that contain no valid data after unstitching.

    This function removes patches that have only missing labels (or only in border
    regions). Useful for training with sparse 3D annotations.

    Parameters
    ----------
    tensor_in : torch.Tensor
        Input patches tensor, typically of shape (N, C, Z, Y, X) or (N, Z, Y, X)
    tensor_out : torch.Tensor
        Output patches tensor with labels, shape (N, C, Z, Y, X) or (N, Z, Y, X).
        Missing/invalid data should be marked with `missing_label`.
    missing_label : Union[int, float]
        Label value that indicates missing/invalid data (typically -1)
    border_tensor : torch.Tensor
        Border mask tensor from `NCZYXQuilt.border_tensor()`.
        Shape should be (Z, Y, X).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple of (valid_input, valid_output, removal_mask) where:
        - valid_input: Filtered input patches (only patches with valid data)
        - valid_output: Filtered output patches (only patches with valid data)
        - removal_mask: Boolean tensor indicating which patches were removed

    Examples
    --------
    >>> from qlty import NCZYXQuilt, weed_sparse_classification_training_pairs_3D
    >>> quilt = NCZYXQuilt(Z=64, Y=64, X=64, window=(32, 32, 32), step=(16, 16, 16), border=(4, 4, 4))
    >>> input_patches = torch.randn(100, 1, 32, 32, 32)
    >>> label_patches = torch.ones(100, 32, 32, 32) * (-1)  # All missing
    >>> label_patches[0:50] = 1.0  # Some valid
    >>> border_tensor = quilt.border_tensor()
    >>> valid_in, valid_out, mask = weed_sparse_classification_training_pairs_3D(
    ...     input_patches, label_patches, missing_label=-1, border_tensor=border_tensor
    ... )
    >>> print(f"Kept {valid_in.shape[0]} out of {input_patches.shape[0]} patches")
    """

    tmp = torch.clone(tensor_out)
    sel = (tmp != missing_label).type(torch.int)

    # Expand border_tensor to match tensor_out shape if needed
    if len(border_tensor.shape) == 3 and len(tensor_out.shape) == 5:
        # tensor_out has channels, expand border_tensor
        border_tensor = border_tensor.unsqueeze(0).unsqueeze(0)
        sel = sel * border_tensor
        sel = einops.reduce(sel, "N C Z Y X -> N", reduction="sum")
    elif len(border_tensor.shape) == 3:
        # tensor_out is (N, Z, Y, X)
        border_tensor = border_tensor.unsqueeze(0)
        sel = sel * border_tensor
        sel = einops.reduce(sel, "N Z Y X -> N", reduction="sum")
    elif len(border_tensor.shape) == 4:
        # tensor_out is (N, C, Z, Y, X) or (N, Z, Y, X)
        border_tensor = border_tensor.unsqueeze(0)
        sel = sel * border_tensor
        if len(tensor_out.shape) == 5:
            sel = einops.reduce(sel, "N C Z Y X -> N", reduction="sum")
        else:
            sel = einops.reduce(sel, "N Z Y X -> N", reduction="sum")
    elif len(border_tensor.shape) == 5:
        sel = sel * border_tensor
        sel = einops.reduce(sel, "N C Z Y X -> N", reduction="sum")
    else:
        # Fallback: multiply and reduce
        sel = sel * border_tensor
        sel = sel.sum(dim=tuple(range(1, len(sel.shape))))

    sel = sel == 0
    newin = tensor_in[~sel, ...]
    newout = tensor_out[~sel, ...]
    return newin, newout, sel
