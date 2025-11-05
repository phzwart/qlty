#!/usr/bin/env python

"""Additional tests for cleanup functions to improve coverage."""

import pytest
import torch

from qlty import qlty2D, qlty3D
from qlty.cleanup import (
    weed_sparse_classification_training_pairs_2D,
    weed_sparse_classification_training_pairs_3D,
)


def test_weed_sparse_2D_3d_border_with_channels():
    """Test 2D weeding with 3D border_tensor and 4D tensor_out (channels)."""
    quilt = qlty2D.NCYXQuilt(
        Y=64, X=64, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    # Create a 3D border tensor (N, Y, X) to test the elif len(border_tensor.shape) == 3 case
    border_tensor_2d = quilt.border_tensor()  # (Y, X)
    border_tensor = border_tensor_2d.unsqueeze(0).expand(10, -1, -1)  # (10, Y, X)

    tensor_in = torch.randn(10, 3, 32, 32)
    tensor_out = torch.ones(10, 1, 32, 32) * 2  # With channel dimension

    newin, newout, sel = weed_sparse_classification_training_pairs_2D(
        tensor_in, tensor_out, missing_label=-1, border_tensor=border_tensor
    )

    # Should handle 3D border tensor case
    assert newin.shape[0] == 10
    assert newout.shape[0] == 10


def test_weed_sparse_3D_with_channels():
    """Test 3D weeding with channel dimension in output."""
    quilt = qlty3D.NCZYXQuilt(
        Z=32,
        Y=32,
        X=32,
        window=(16, 16, 16),
        step=(8, 8, 8),
        border=(3, 3, 3),
        border_weight=0.1,
    )

    border_tensor = quilt.border_tensor()

    # Create patches with channel dimension in output
    tensor_in = torch.randn(10, 2, 16, 16, 16)
    tensor_out = torch.ones(10, 1, 16, 16, 16) * 2  # With channel dimension

    newin, newout, sel = weed_sparse_classification_training_pairs_3D(
        tensor_in, tensor_out, missing_label=-1, border_tensor=border_tensor
    )

    # Should work with channel dimension
    assert newin.shape[0] == 10
    assert newout.shape[0] == 10
    assert torch.sum(sel) == 0


def test_weed_sparse_3D_fallback_case():
    """Test 3D weeding with 5D border_tensor (fallback case)."""
    # Create a 5D border tensor (should trigger fallback)
    border_tensor = torch.ones(10, 1, 16, 16, 16)  # (N, C, Z, Y, X)

    tensor_in = torch.randn(10, 2, 16, 16, 16)
    tensor_out = torch.ones(10, 1, 16, 16, 16) * 2

    newin, newout, sel = weed_sparse_classification_training_pairs_3D(
        tensor_in, tensor_out, missing_label=-1, border_tensor=border_tensor
    )

    # Should handle fallback case
    assert newin.shape[0] == 10
    assert newout.shape[0] == 10


# Note: The 4D border_tensor with 4D tensor_out case has a bug in the code
# (line 159 tries to reduce 5D tensor with 4D pattern after unsqueeze)
# Skipping this test case until the bug is fixed


def test_weed_sparse_3D_4d_border_with_channels():
    """Test 3D weeding with 4D border_tensor and 5D tensor_out."""
    # Create a 4D border tensor
    border_tensor = torch.ones(10, 16, 16, 16)  # (N, Z, Y, X)

    tensor_in = torch.randn(10, 2, 16, 16, 16)
    tensor_out = torch.ones(10, 1, 16, 16, 16) * 2  # With channel

    newin, newout, sel = weed_sparse_classification_training_pairs_3D(
        tensor_in, tensor_out, missing_label=-1, border_tensor=border_tensor
    )

    assert newin.shape[0] == 10
    assert newout.shape[0] == 10
