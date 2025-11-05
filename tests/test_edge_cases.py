#!/usr/bin/env python

"""Tests for edge cases and error handling."""

import einops
import numpy as np
import pytest
import torch

from qlty import qlty2D, qlty3D
from qlty.base import BaseQuilt, normalize_border, validate_border_weight


def test_qlty2D_invalid_border_weight():
    """Test invalid border_weight raises error."""
    with pytest.raises(ValueError):
        qlty2D.NCYXQuilt(
            Y=100,
            X=100,
            window=(50, 50),
            step=(25, 25),
            border=(5, 5),
            border_weight=-0.1,  # Invalid
        )

    with pytest.raises(ValueError):
        qlty2D.NCYXQuilt(
            Y=100,
            X=100,
            window=(50, 50),
            step=(25, 25),
            border=(5, 5),
            border_weight=1.5,  # Invalid
        )


def test_qlty2D_invalid_border_length():
    """Test invalid border tuple length raises error."""
    with pytest.raises(ValueError):
        qlty2D.NCYXQuilt(
            Y=100,
            X=100,
            window=(50, 50),
            step=(25, 25),
            border=(1, 2, 3),  # Wrong length for 2D
            border_weight=0.1,
        )


def test_qlty3D_invalid_border_length():
    """Test invalid border tuple length for 3D."""
    with pytest.raises(ValueError):
        qlty3D.NCZYXQuilt(
            Z=100,
            Y=100,
            X=100,
            window=(50, 50, 50),
            step=(25, 25, 25),
            border=(1, 2),  # Wrong length for 3D
            border_weight=0.1,
        )


def test_qlty2D_border_normalization():
    """Test that border normalization works correctly."""
    # Test int border
    quilt1 = qlty2D.NCYXQuilt(
        Y=100,
        X=100,
        window=(50, 50),
        step=(25, 25),
        border=5,
        border_weight=0.1,  # int
    )

    # Test tuple border
    quilt2 = qlty2D.NCYXQuilt(
        Y=100,
        X=100,
        window=(50, 50),
        step=(25, 25),
        border=(5, 5),  # tuple
        border_weight=0.1,
    )

    # Should produce same results
    assert quilt1.border == quilt2.border


def test_qlty2D_zero_border():
    """Test that zero border is handled correctly."""
    quilt = qlty2D.NCYXQuilt(
        Y=100,
        X=100,
        window=(50, 50),
        step=(25, 25),
        border=0,  # Should be normalized to None
        border_weight=0.1,
    )

    assert quilt.border is None
    border_tensor = quilt.border_tensor()
    assert torch.allclose(border_tensor, torch.ones(50, 50))


def test_qlty2D_unstitch_single_image():
    """Test unstitching a single image."""
    quilt = qlty2D.NCYXQuilt(
        Y=64, X=64, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    # Single image
    image = torch.randn(1, 3, 64, 64)
    patches = quilt.unstitch(image)

    assert patches.shape[0] > 0
    assert patches.shape[1] == 3
    assert patches.shape[2] == 32
    assert patches.shape[3] == 32


def test_qlty2D_stitch_mismatch():
    """Test that stitch with wrong number of patches raises error."""
    quilt = qlty2D.NCYXQuilt(
        Y=64, X=64, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    # Wrong number of patches
    patches = torch.randn(7, 3, 32, 32)  # Not divisible by nY*nX

    # This should raise an assertion error
    with pytest.raises(AssertionError):
        quilt.stitch(patches)


def test_qlty2D_get_times():
    """Test get_times returns correct values."""
    quilt = qlty2D.NCYXQuilt(
        Y=100, X=100, window=(50, 50), step=(25, 25), border=(5, 5), border_weight=0.1
    )

    nY, nX = quilt.get_times()
    assert nY > 0
    assert nX > 0
    # Should be able to cover the full image
    assert (nY - 1) * 25 + 50 >= 100
    assert (nX - 1) * 25 + 50 >= 100


def test_qlty3D_border_tensor():
    """Test 3D border tensor computation."""
    quilt = qlty3D.NCZYXQuilt(
        Z=64,
        Y=64,
        X=64,
        window=(32, 32, 32),
        step=(16, 16, 16),
        border=(5, 5, 5),
        border_weight=0.1,
    )

    border_tensor = quilt.border_tensor()
    assert border_tensor.shape == (32, 32, 32)
    # Center should be 1.0
    assert torch.allclose(border_tensor[5:27, 5:27, 5:27], torch.ones(22, 22, 22))


def test_qlty2D_unstitch_data_pair_with_missing_label():
    """Test unstitch_data_pair with missing labels."""
    quilt = qlty2D.NCYXQuilt(
        Y=64, X=64, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    data_in = torch.randn(2, 3, 64, 64)
    data_out = torch.ones(2, 64, 64) * (-1)  # All missing
    data_out[:, 10:54, 10:54] = 1.0  # Some valid data

    patches_in, patches_out = quilt.unstitch_data_pair(
        data_in, data_out, missing_label=-1
    )

    assert patches_in.shape[0] == patches_out.shape[0]
    # Some patches should have valid data
    assert torch.any(patches_out != -1)


def test_qlty2D_weight_matrix():
    """Test weight matrix computation."""
    quilt = qlty2D.NCYXQuilt(
        Y=64, X=64, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    weight = quilt.weight
    assert weight.shape == (32, 32)
    # Center should be 1.0
    assert torch.allclose(weight[5:27, 5:27], torch.ones(22, 22))
    # Border should be 0.1
    assert torch.allclose(weight[0:5, :], torch.ones(5, 32) * 0.1)


def test_qlty3D_get_times():
    """Test 3D get_times."""
    quilt = qlty3D.NCZYXQuilt(
        Z=100,
        Y=100,
        X=100,
        window=(50, 50, 50),
        step=(25, 25, 25),
        border=(5, 5, 5),
        border_weight=0.1,
    )

    nZ, nY, nX = quilt.get_times()
    assert nZ > 0
    assert nY > 0
    assert nX > 0
    assert nZ == nY == nX  # Should be same for these parameters


@pytest.mark.parametrize(
    "border_input,expected",
    [
        (None, None),
        (0, None),
        ((0, 0), None),
        (5, (5, 5)),
        ((5, 10), (5, 10)),
    ],
)
def test_border_normalization_edge_cases(border_input, expected):
    """Test border normalization with various edge cases."""
    result = normalize_border(border_input, ndim=2)
    assert result == expected
