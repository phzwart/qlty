#!/usr/bin/env python

"""Additional tests for qlty2DLarge to improve coverage."""

import os
import tempfile

import numpy as np
import pytest
import torch

from qlty import qlty2DLarge


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture to create a temporary directory for Zarr files."""
    path = tmp_path / "zarr_test"
    path.mkdir()
    yield str(path)
    # Cleanup
    for suffix in [
        "_mean_cache.zarr",
        "_std_cache.zarr",
        "_norma_cache.zarr",
        "_mean.zarr",
        "_std.zarr",
    ]:
        zarr_path = os.path.join(path, f"test{suffix}")
        if os.path.exists(zarr_path):
            import shutil

            shutil.rmtree(zarr_path)


def test_return_mean_with_normalize(temp_dir):
    """Test return_mean with normalize=True."""
    filename = os.path.join(temp_dir, "test_normalize")
    N, C, Y, X = 2, 3, 64, 64
    window = (32, 32)
    step = (16, 16)
    border = (5, 5)

    quilt = qlty2DLarge.LargeNCYXQuilt(
        filename=filename,
        N=N,
        Y=Y,
        X=X,
        window=window,
        step=step,
        border=border,
        border_weight=0.1,
    )

    data = torch.randn(N, C, Y, X)

    # Process all chunks
    for _ in range(quilt.N_chunks):
        idx, patch = quilt.unstitch_next(data)
        processed = patch.unsqueeze(0)  # Add batch dimension
        quilt.stitch(processed, idx)

    # Get normalized mean
    mean_normalized = quilt.return_mean(normalize=True)

    assert mean_normalized.shape == (N, C, Y, X)
    # Normalize=True divides by sum along axis=0 (batch dimension)
    # Check that values are finite
    assert np.isfinite(mean_normalized).all()
    # After normalization, sum along axis=0 should equal 1 (or close to it)
    # But only check if sum is not zero to avoid division by zero issues
    sum_along_batch = np.sum(mean_normalized, axis=0)
    non_zero_mask = np.abs(sum_along_batch) > 1e-10
    if np.any(non_zero_mask):
        # Where sum is non-zero, normalized values should have sum ≈ 1
        normalized_sums = sum_along_batch[non_zero_mask]
        # Allow some tolerance for numerical precision
        assert np.allclose(normalized_sums, 1.0, atol=1e-6, rtol=1e-6)


def test_return_mean_normalize_with_std(temp_dir):
    """Test return_mean with normalize=True and std=True."""
    filename = os.path.join(temp_dir, "test_normalize_std")
    N, C, Y, X = 2, 3, 64, 64
    window = (32, 32)
    step = (16, 16)
    border = (5, 5)

    quilt = qlty2DLarge.LargeNCYXQuilt(
        filename=filename,
        N=N,
        Y=Y,
        X=X,
        window=window,
        step=step,
        border=border,
        border_weight=0.1,
    )

    data = torch.randn(N, C, Y, X)

    # Process all chunks with variance
    for _ in range(quilt.N_chunks):
        idx, patch = quilt.unstitch_next(data)
        processed = patch.unsqueeze(0)
        patch_var = torch.var(processed, dim=1, keepdim=True)  # Variance per channel
        quilt.stitch(processed, idx, patch_var=patch_var)

    # Get normalized mean and std
    mean_norm, std_norm = quilt.return_mean(std=True, normalize=True)

    assert mean_norm.shape == (N, C, Y, X)
    assert std_norm.shape == (N, C, Y, X)
    # Check that values are finite
    assert np.isfinite(mean_norm).all()
    assert np.isfinite(std_norm).all()
    # After normalization, sum along axis=0 should equal 1 (or close to it)
    # But only check if sum is not zero to avoid division by zero issues
    sum_along_batch = np.sum(mean_norm, axis=0)
    non_zero_mask = np.abs(sum_along_batch) > 1e-10
    if np.any(non_zero_mask):
        # Where sum is non-zero, normalized values should have sum ≈ 1
        normalized_sums = sum_along_batch[non_zero_mask]
        # Allow some tolerance for numerical precision
        assert np.allclose(normalized_sums, 1.0, atol=1e-6, rtol=1e-6)


def test_stitch_with_patch_var(temp_dir):
    """Test stitch method with patch_var parameter."""
    filename = os.path.join(temp_dir, "test_var")
    N, C, Y, X = 2, 3, 64, 64
    window = (32, 32)
    step = (16, 16)

    quilt = qlty2DLarge.LargeNCYXQuilt(
        filename=filename,
        N=N,
        Y=Y,
        X=X,
        window=window,
        step=step,
        border=(5, 5),
        border_weight=0.1,
    )

    data = torch.randn(N, C, Y, X)

    # Process with variance
    for _ in range(quilt.N_chunks):
        idx, patch = quilt.unstitch_next(data)
        processed = patch.unsqueeze(0)
        patch_var = torch.var(processed, dim=1, keepdim=True)
        quilt.stitch(processed, idx, patch_var=patch_var)

    # Get mean and std
    mean, std = quilt.return_mean(std=True)

    assert mean.shape == (N, C, Y, X)
    assert std.shape == (N, C, Y, X)
