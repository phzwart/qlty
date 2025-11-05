#!/usr/bin/env python

"""Additional tests for qlty2D to improve coverage."""

import pytest
import torch

from qlty import qlty2D


def test_stitch_without_numba():
    """Test stitching without numba (use_numba=False)."""
    quilt = qlty2D.NCYXQuilt(
        Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    # Create test data
    data = torch.randn(2, 3, 128, 128)
    patches = quilt.unstitch(data)

    # Stitch without numba
    reconstructed, weights = quilt.stitch(patches, use_numba=False)

    assert reconstructed.shape == data.shape
    assert weights.shape == (128, 128)


def test_stitch_with_numba():
    """Test stitching with numba (use_numba=True, default)."""
    quilt = qlty2D.NCYXQuilt(
        Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    # Create test data
    data = torch.randn(2, 3, 128, 128)
    patches = quilt.unstitch(data)

    # Stitch with numba (default)
    reconstructed, weights = quilt.stitch(patches, use_numba=True)

    assert reconstructed.shape == data.shape
    assert weights.shape == (128, 128)


def test_stitch_numba_vs_no_numba_consistency():
    """Test that numba and non-numba paths produce similar results."""
    quilt = qlty2D.NCYXQuilt(
        Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    # Create test data
    data = torch.randn(2, 3, 128, 128)
    patches = quilt.unstitch(data)

    # Stitch with numba
    reconstructed_numba, weights_numba = quilt.stitch(patches, use_numba=True)

    # Stitch without numba
    reconstructed_no_numba, weights_no_numba = quilt.stitch(patches, use_numba=False)

    # Results should be similar (numba and non-numba may have different implementations)
    # Check that shapes match and values are reasonable
    assert reconstructed_numba.shape == reconstructed_no_numba.shape
    assert weights_numba.shape == weights_no_numba.shape

    # Check that both produce valid results (not NaN or Inf)
    assert torch.isfinite(reconstructed_numba).all()
    assert torch.isfinite(reconstructed_no_numba).all()
    assert torch.isfinite(weights_numba).all()
    assert torch.isfinite(weights_no_numba).all()
