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
