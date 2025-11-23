#!/usr/bin/env python

"""Tests for base utilities."""

import numpy as np
import pytest
import torch

from qlty.base import (
    compute_border_tensor_numpy,
    compute_border_tensor_torch,
    compute_chunk_times,
    compute_weight_matrix_numpy,
    compute_weight_matrix_torch,
    normalize_border,
    validate_border_weight,
)


def test_normalize_border():
    """Test border normalization."""
    # Test None
    assert normalize_border(None, 2) is None
    assert normalize_border(None, 3) is None

    # Test int
    assert normalize_border(0, 2) is None
    assert normalize_border(5, 2) == (5, 5)
    assert normalize_border(3, 3) == (3, 3, 3)

    # Test tuple
    assert normalize_border((0, 0), 2) is None
    assert normalize_border((5, 10), 2) == (5, 10)
    assert normalize_border((1, 2, 3), 3) == (1, 2, 3)

    # Test invalid inputs
    with pytest.raises(ValueError):
        normalize_border((1, 2), 3)  # Wrong length

    with pytest.raises(TypeError):
        normalize_border("invalid", 2)


def test_validate_border_weight():
    """Test border weight validation."""
    # Valid weights
    assert validate_border_weight(0.0) == 1e-8
    assert validate_border_weight(0.5) == 0.5
    assert validate_border_weight(1.0) == 1.0
    assert validate_border_weight(0.1) == 0.1

    # Invalid weights
    with pytest.raises(ValueError):
        validate_border_weight(-0.1)

    with pytest.raises(ValueError):
        validate_border_weight(1.5)

    with pytest.raises(ValueError):
        validate_border_weight(2.0)


def test_compute_weight_matrix_torch():
    """Test weight matrix computation (torch)."""
    # No border
    weight = compute_weight_matrix_torch((10, 10), None, 0.1)
    assert weight.shape == (10, 10)
    assert torch.allclose(weight, torch.ones(10, 10))

    # With border
    weight = compute_weight_matrix_torch((10, 10), (2, 2), 0.1)
    assert weight.shape == (10, 10)
    # Center should be 1.0
    assert torch.allclose(weight[2:8, 2:8], torch.ones(6, 6))
    # Border should be 0.1
    assert torch.allclose(weight[0:2, :], torch.ones(2, 10) * 0.1)
    assert torch.allclose(weight[:, 0:2], torch.ones(10, 2) * 0.1)


def test_compute_weight_matrix_numpy():
    """Test weight matrix computation (numpy)."""
    # No border
    weight = compute_weight_matrix_numpy((10, 10), None, 0.1)
    assert weight.shape == (10, 10)
    assert np.allclose(weight, np.ones((10, 10)) * 0.1)

    # With border
    weight = compute_weight_matrix_numpy((10, 10), (2, 2), 0.1)
    assert weight.shape == (10, 10)
    # Center should be 1.0
    assert np.allclose(weight[2:8, 2:8], np.ones((6, 6)))
    # Border should be 0.1
    assert np.allclose(weight[0:2, :], np.ones((2, 10)) * 0.1)


def test_compute_border_tensor_torch():
    """Test border tensor computation (torch)."""
    # No border
    border_tensor = compute_border_tensor_torch((10, 10), None)
    assert border_tensor.shape == (10, 10)
    assert torch.allclose(border_tensor, torch.ones(10, 10))

    # With border
    border_tensor = compute_border_tensor_torch((10, 10), (2, 2))
    assert border_tensor.shape == (10, 10)
    # Center should be 1.0
    assert torch.allclose(border_tensor[2:8, 2:8], torch.ones(6, 6))
    # Border should be 0.0
    assert torch.allclose(border_tensor[0:2, :], torch.zeros(2, 10))
    assert torch.allclose(border_tensor[:, 0:2], torch.zeros(10, 2))


def test_compute_border_tensor_numpy():
    """Test border tensor computation (numpy)."""
    # No border
    border_tensor = compute_border_tensor_numpy((10, 10), None)
    assert border_tensor.shape == (10, 10)
    assert np.allclose(border_tensor, np.ones((10, 10)))

    # With border
    border_tensor = compute_border_tensor_numpy((10, 10), (2, 2))
    assert border_tensor.shape == (10, 10)
    # Center should be 1.0
    assert np.allclose(border_tensor[2:8, 2:8], np.ones((6, 6)))
    # Border should be 0.0
    assert np.allclose(border_tensor[0:2, :], np.zeros((2, 10)))
    assert np.allclose(border_tensor[:, 0:2], np.zeros((10, 2)))


def test_compute_chunk_times():
    """Test chunk times computation."""
    # Simple case
    times = compute_chunk_times((100, 100), (50, 50), (25, 25))
    assert times == (3, 3)  # 0, 25, 50, 75, 100 (but 75+50 > 100, so adjust)

    # Edge case: exact fit
    times = compute_chunk_times((100, 100), (50, 50), (50, 50))
    assert times == (2, 2)  # 0, 50, 100

    # 3D case
    times = compute_chunk_times((64, 64, 64), (32, 32, 32), (16, 16, 16))
    assert times == (3, 3, 3)

    # Unequal dimensions
    times = compute_chunk_times((100, 50), (30, 20), (20, 10))
    assert len(times) == 2
    assert times[0] >= 1
    assert times[1] >= 1


def test_compute_chunk_times_edge_cases():
    """Test chunk times with edge cases."""
    # Window larger than step
    times = compute_chunk_times((100, 100), (60, 60), (20, 20))
    assert times[0] >= 3  # Should have at least a few chunks

    # Step larger than dimension (should still work)
    times = compute_chunk_times((50, 50), (30, 30), (40, 40))
    assert times[0] >= 1
    assert times[1] >= 1


def test_compute_weight_matrix_torch_with_zero_border():
    """Test weight matrix with border containing zero values."""
    # Border with zero in one dimension
    weight = compute_weight_matrix_torch((10, 10), (2, 0), 0.1)
    assert weight.shape == (10, 10)
    # Should handle zero border value correctly


def test_compute_border_tensor_torch_with_zero_border():
    """Test border tensor with border containing zero values."""
    # Border with zero in one dimension
    border_tensor = compute_border_tensor_torch((10, 10), (2, 0))
    assert border_tensor.shape == (10, 10)
    # Should handle zero border value correctly


def test_compute_weight_matrix_numpy_with_zero_border():
    """Test weight matrix (numpy) with border containing zero values."""
    # Border with zero in one dimension
    weight = compute_weight_matrix_numpy((10, 10), (2, 0), 0.1)
    assert weight.shape == (10, 10)
    # Should handle zero border value correctly


def test_compute_border_tensor_numpy_with_zero_border():
    """Test border tensor (numpy) with border containing zero values."""
    # Border with zero in one dimension
    border_tensor = compute_border_tensor_numpy((10, 10), (2, 0))
    assert border_tensor.shape == (10, 10)
    # Should handle zero border value correctly


def test_base_quilt_validation():
    """Test BaseQuilt validation logic."""
    from qlty.base import BaseQuilt

    # Valid initialization (2D)
    quilt = BaseQuilt(
        window=(32, 32),
        step=(16, 16),
        border=(5, 5),
        border_weight=0.1,
        ndim=2,
    )
    assert quilt.window == (32, 32)
    assert quilt.step == (16, 16)
    assert quilt.border == (5, 5)
    assert quilt.ndim == 2

    # Valid initialization (3D)
    quilt3d = BaseQuilt(
        window=(32, 32, 32),
        step=(16, 16, 16),
        border=(5, 5, 5),
        border_weight=0.1,
        ndim=3,
    )
    assert quilt3d.ndim == 3

    # Invalid window length
    with pytest.raises(ValueError, match="window must have 2 elements"):
        BaseQuilt(
            window=(32, 32, 32),  # Wrong length for 2D
            step=(16, 16),
            border=None,
            border_weight=0.1,
            ndim=2,
        )

    # Invalid step length
    with pytest.raises(ValueError, match="step must have 3 elements"):
        BaseQuilt(
            window=(32, 32, 32),
            step=(16, 16),  # Wrong length for 3D
            border=None,
            border_weight=0.1,
            ndim=3,
        )

    # Invalid border length - this is caught by normalize_border first
    # So we need to test the case where normalize_border returns a border
    # but it has the wrong length after normalization
    # Actually, normalize_border already validates length, so this case is unreachable
    # But we can test it directly by bypassing normalize_border
    with pytest.raises(ValueError, match="border tuple must have 2 elements"):
        # This raises in normalize_border, so BaseQuilt.__init__ line 305 is never reached
        BaseQuilt(
            window=(32, 32),
            step=(16, 16),
            border=(5, 5, 5),  # Wrong length for 2D
            border_weight=0.1,
            ndim=2,
        )

    # To reach line 305, we'd need normalize_border to succeed but return wrong length
    # But that's impossible since normalize_border validates length.
    # Line 305 appears to be defensive code for an unreachable case.
