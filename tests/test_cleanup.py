#!/usr/bin/env python

"""Tests for cleanup functions."""

import einops
import pytest
import torch

from qlty import qlty2D, qlty3D
from qlty.cleanup import (
    weed_sparse_classification_training_pairs_2D,
    weed_sparse_classification_training_pairs_3D,
)


def test_weed_sparse_2D_all_valid():
    """Test weeding with all valid patches."""
    quilt = qlty2D.NCYXQuilt(
        Y=64, X=64, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    border_tensor = quilt.border_tensor()

    # Create patches with all valid data
    tensor_in = torch.randn(10, 3, 32, 32)
    tensor_out = torch.ones(10, 32, 32) * 2  # All valid (not missing_label)

    newin, newout, sel = weed_sparse_classification_training_pairs_2D(
        tensor_in, tensor_out, missing_label=-1, border_tensor=border_tensor
    )

    # Should keep all patches
    assert newin.shape[0] == 10
    assert newout.shape[0] == 10
    assert torch.sum(sel) == 0  # None selected for removal


def test_weed_sparse_2D_all_missing():
    """Test weeding with all missing patches."""
    quilt = qlty2D.NCYXQuilt(
        Y=64, X=64, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    border_tensor = quilt.border_tensor()

    # Create patches with all missing data
    tensor_in = torch.randn(10, 3, 32, 32)
    tensor_out = torch.ones(10, 32, 32) * (-1)  # All missing

    newin, newout, sel = weed_sparse_classification_training_pairs_2D(
        tensor_in, tensor_out, missing_label=-1, border_tensor=border_tensor
    )

    # Should remove all patches (only border areas have data, but border is masked)
    assert newin.shape[0] == 0
    assert newout.shape[0] == 0
    assert torch.sum(sel) == 10  # All selected for removal


def test_weed_sparse_2D_mixed():
    """Test weeding with mixed valid/missing patches."""
    quilt = qlty2D.NCYXQuilt(
        Y=64, X=64, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    border_tensor = quilt.border_tensor()

    # Create patches with some valid data
    tensor_in = torch.randn(10, 3, 32, 32)
    tensor_out = torch.ones(10, 32, 32) * (-1)  # Start with all missing

    # Add valid data to center region for some patches
    valid_mask = border_tensor > 0.5
    for i in range(5):  # First 5 patches have valid data
        tensor_out[i, valid_mask] = 1.0

    newin, newout, sel = weed_sparse_classification_training_pairs_2D(
        tensor_in, tensor_out, missing_label=-1, border_tensor=border_tensor
    )

    # Should keep patches with valid data
    assert newin.shape[0] > 0
    assert newout.shape[0] > 0
    assert newin.shape[0] == newout.shape[0]
    assert torch.sum(sel) < 10  # Some removed


def test_weed_sparse_3D_all_valid():
    """Test 3D weeding with all valid patches."""
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

    # Create patches with all valid data
    tensor_in = torch.randn(10, 2, 16, 16, 16)
    tensor_out = torch.ones(10, 16, 16, 16) * 2  # All valid

    newin, newout, sel = weed_sparse_classification_training_pairs_3D(
        tensor_in, tensor_out, missing_label=-1, border_tensor=border_tensor
    )

    # Should keep all patches
    assert newin.shape[0] == 10
    assert newout.shape[0] == 10
    assert torch.sum(sel) == 0


def test_weed_sparse_3D_all_missing():
    """Test 3D weeding with all missing patches."""
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

    # Create patches with all missing data
    tensor_in = torch.randn(10, 2, 16, 16, 16)
    tensor_out = torch.ones(10, 16, 16, 16) * (-1)  # All missing

    newin, newout, sel = weed_sparse_classification_training_pairs_3D(
        tensor_in, tensor_out, missing_label=-1, border_tensor=border_tensor
    )

    # Should remove all patches
    assert newin.shape[0] == 0
    assert newout.shape[0] == 0
    assert torch.sum(sel) == 10


def test_weed_sparse_2D_with_channels():
    """Test 2D weeding with channel dimension."""
    quilt = qlty2D.NCYXQuilt(
        Y=64, X=64, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    border_tensor = quilt.border_tensor()

    # Create patches with channel dimension in output
    tensor_in = torch.randn(10, 3, 32, 32)
    tensor_out = torch.ones(10, 1, 32, 32) * 2  # With channel dimension

    newin, newout, sel = weed_sparse_classification_training_pairs_2D(
        tensor_in, tensor_out, missing_label=-1, border_tensor=border_tensor
    )

    # Should work with channel dimension
    assert newin.shape[0] == 10
    assert newout.shape[0] == 10
    assert torch.sum(sel) == 0  # None selected for removal
