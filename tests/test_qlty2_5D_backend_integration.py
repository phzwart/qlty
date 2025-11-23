"""
Integration tests for 2.5D Quilt with backends.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from qlty.backends_2_5D import InMemoryBackend, TensorLike3D, ZarrBackend
from qlty.qlty2_5D import NCZYX25DQuilt


def test_qlty_with_tensor_like_in_memory():
    """Test NCZYX25DQuilt with TensorLike3D wrapper."""
    # Create test data
    data = torch.randn(1, 1, 10, 20, 20)

    # Wrap in backend and tensor-like
    backend = InMemoryBackend(data)
    tensor_like = TensorLike3D(backend)

    # Create quilt
    spec = {"identity": [0]}
    quilt = NCZYX25DQuilt(
        data_source=tensor_like, channel_spec=spec, accumulation_mode="2d", z_slices=[0]
    )

    # Convert
    result = quilt.convert()

    assert result.shape == (1, 1, 20, 20)
    assert torch.allclose(result[0, 0], data[0, 0, 0])


def test_qlty_with_tensor_like_zarr():
    """Test NCZYX25DQuilt with ZarrBackend."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    # Create zarr array
    shape = (1, 1, 10, 20, 20)
    z = zarr.zeros(shape, dtype="float32")
    data = np.random.randn(*shape).astype(np.float32)
    z[:] = data[:]

    # Wrap in backend and tensor-like
    backend = ZarrBackend(z)
    tensor_like = TensorLike3D(backend)

    # Create quilt
    spec = {"identity": [-1, 0, 1]}
    quilt = NCZYX25DQuilt(
        data_source=tensor_like,
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[5],  # Center at z=5
    )

    # Convert
    result = quilt.convert()

    # Should have 3 channels (from direct [-1, 0, 1])
    assert result.shape == (1, 3, 20, 20)


def test_qlty_with_direct_tensor():
    """Test that direct torch.Tensor still works."""
    data = torch.randn(1, 1, 10, 20, 20)

    spec = {"identity": [0]}
    quilt = NCZYX25DQuilt(
        data_source=data,  # Direct tensor, not wrapped
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[0],
    )

    result = quilt.convert()
    assert result.shape == (1, 1, 20, 20)


if __name__ == "__main__":
    test_qlty_with_tensor_like_in_memory()
    print("✓ qlty_with_tensor_like_in_memory")

    test_qlty_with_tensor_like_zarr()
    print("✓ qlty_with_tensor_like_zarr")

    test_qlty_with_direct_tensor()
    print("✓ qlty_with_direct_tensor")

    print("\nAll integration tests passed!")
