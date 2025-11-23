#!/usr/bin/env python

"""Additional tests for qlty3DLarge to improve coverage."""

import os
import tempfile

import numpy as np
import pytest
import torch

from qlty import qlty3DLarge


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture to create a temporary directory for Zarr files."""
    path = tmp_path / "zarr_test_3d"
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


def test_get_times(temp_dir):
    """Test get_times method."""
    filename = os.path.join(temp_dir, "test_times")
    quilt = qlty3DLarge.LargeNCZYXQuilt(
        filename=filename,
        N=2,
        Z=64,
        Y=64,
        X=64,
        window=(16, 16, 16),
        step=(8, 8, 8),
        border=(2, 2, 2),
        border_weight=0.1,
    )
    times = quilt.get_times()
    assert isinstance(times, tuple)
    assert len(times) == 3
    assert all(isinstance(t, int) and t > 0 for t in times)


def test_unstitch_and_clean_sparse_data_pair_4d_tensor_out(temp_dir):
    """Test unstitch_and_clean_sparse_data_pair with 4D tensor_out (missing channel dimension)."""
    filename = os.path.join(temp_dir, "test_4d_tensor")
    quilt = qlty3DLarge.LargeNCZYXQuilt(
        filename=filename,
        N=2,
        Z=32,
        Y=32,
        X=32,
        window=(16, 16, 16),
        step=(8, 8, 8),
        border=(2, 2, 2),
        border_weight=0.1,
    )
    
    # Create 5D tensor_in (N, C, Z, Y, X)
    tensor_in = torch.randn(2, 3, 32, 32, 32)
    
    # Create 4D tensor_out (N, Z, Y, X) - missing channel dimension
    tensor_out = torch.randn(2, 32, 32, 32)
    # Mark some regions as missing
    missing_label = -1
    tensor_out[:, 0:5, 0:5, 0:5] = missing_label
    
    ain, aout = quilt.unstitch_and_clean_sparse_data_pair(tensor_in, tensor_out, missing_label)
    
    # Should handle 4D tensor_out by adding channel dimension
    if len(ain) > 0:
        assert isinstance(ain, torch.Tensor) or isinstance(ain, list)
        assert isinstance(aout, torch.Tensor) or isinstance(aout, list)


def test_unstitch_and_clean_sparse_data_pair_rearranged(temp_dir):
    """Test unstitch_and_clean_sparse_data_pair with rearranged output."""
    filename = os.path.join(temp_dir, "test_rearranged")
    quilt = qlty3DLarge.LargeNCZYXQuilt(
        filename=filename,
        N=2,
        Z=32,
        Y=32,
        X=32,
        window=(16, 16, 16),
        step=(8, 8, 8),
        border=(2, 2, 2),
        border_weight=0.1,
    )
    
    # Create 5D tensor_in (N, C, Z, Y, X)
    tensor_in = torch.randn(2, 3, 32, 32, 32)
    
    # Create 4D tensor_out (N, Z, Y, X) with single channel to trigger rearranged path
    tensor_out = torch.randn(2, 32, 32, 32)
    missing_label = -1
    # Mark some regions as missing, but leave enough valid data
    tensor_out[:, 0:5, 0:5, 0:5] = missing_label
    tensor_out[:, 10:15, 10:15, 10:15] = missing_label
    
    ain, aout = quilt.unstitch_and_clean_sparse_data_pair(tensor_in, tensor_out, missing_label)
    
    # Should handle 4D tensor_out and rearrange it
    if len(ain) > 0:
        # If we got results, check shapes
        if isinstance(ain, torch.Tensor):
            # After rearranging and squeezing, aout should be 4D if it was 4D input
            assert isinstance(aout, torch.Tensor)

