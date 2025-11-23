"""
Tests for 2.5D Quilt backends.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from qlty.backends_2_5D import (
    HDF5Backend,
    InMemoryBackend,
    MemoryMappedBackend,
    TensorLike3D,
    ZarrBackend,
)


def test_in_memory_backend():
    """Test InMemoryBackend with torch.Tensor."""
    data = torch.randn(2, 3, 5, 10, 10)
    backend = InMemoryBackend(data)

    assert backend.get_shape() == (2, 3, 5, 10, 10)
    assert backend.get_dtype() == data.dtype
    assert backend.supports_batch_loading

    # Test loading single slice
    result = backend.load_slice(n=0, c=0, z=2)
    assert result.shape == (10, 10)
    assert torch.allclose(result, data[0, 0, 2])

    # Test loading z range
    result = backend.load_slice(n=0, c=0, z=slice(1, 4))
    assert result.shape == (3, 10, 10)
    assert torch.allclose(result, data[0, 0, 1:4])


def test_tensor_like_3d_in_memory():
    """Test TensorLike3D wrapper with InMemoryBackend."""
    data = torch.randn(2, 3, 5, 10, 10)
    backend = InMemoryBackend(data)
    tensor_like = TensorLike3D(backend)

    assert tensor_like.shape == (2, 3, 5, 10, 10)
    assert tensor_like.dtype == data.dtype
    assert len(tensor_like) == 2

    # Test indexing
    result = tensor_like[0]
    assert result.shape == (3, 5, 10, 10)
    assert torch.allclose(result, data[0])

    result = tensor_like[0, 1, 2]
    assert result.shape == (10, 10)
    assert torch.allclose(result, data[0, 1, 2])

    result = tensor_like[0, 1, 1:4]
    assert result.shape == (3, 10, 10)
    assert torch.allclose(result, data[0, 1, 1:4])


def test_memory_mapped_backend():
    """Test MemoryMappedBackend with numpy memmap."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name

    try:
        # Create memory-mapped array
        shape = (2, 3, 5, 10, 10)
        data = np.random.randn(*shape).astype(np.float32)
        mmap = np.memmap(temp_path, dtype="float32", mode="w+", shape=shape)
        mmap[:] = data[:]
        mmap.flush()

        # Load as read-only
        mmap_read = np.memmap(temp_path, dtype="float32", mode="r", shape=shape)
        backend = MemoryMappedBackend(mmap_read)

        assert backend.get_shape() == shape
        assert backend.supports_batch_loading

        # Test loading
        result = backend.load_slice(n=0, c=0, z=2)
        assert result.shape == (10, 10)
        assert np.allclose(result.numpy(), data[0, 0, 2])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_zarr_backend():
    """Test ZarrBackend with zarr array."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    # Create zarr array
    shape = (2, 3, 5, 10, 10)
    z = zarr.zeros(shape, dtype="float32")
    data = np.random.randn(*shape).astype(np.float32)
    z[:] = data[:]

    backend = ZarrBackend(z)

    assert backend.get_shape() == shape
    assert backend.supports_batch_loading

    # Test loading
    result = backend.load_slice(n=0, c=0, z=2)
    assert result.shape == (10, 10)
    assert np.allclose(result.numpy(), data[0, 0, 2])

    # Test with 4D zarr (C, Z, Y, X)
    z_4d = zarr.zeros((3, 5, 10, 10), dtype="float32")
    data_4d = np.random.randn(3, 5, 10, 10).astype(np.float32)
    z_4d[:] = data_4d[:]

    backend_4d = ZarrBackend(z_4d)
    assert backend_4d.get_shape() == (1, 3, 5, 10, 10)

    result = backend_4d.load_slice(c=0, z=2)
    assert result.shape == (10, 10)
    assert np.allclose(result.numpy(), data_4d[0, 2])


def test_hdf5_backend():
    """Test HDF5Backend with h5py dataset."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    # Create HDF5 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        shape = (2, 3, 5, 10, 10)
        data = np.random.randn(*shape).astype(np.float32)

        with h5py.File(temp_path, "w") as f:
            dset = f.create_dataset("data", shape=shape, dtype="float32")
            dset[:] = data[:]

        # Open and test
        with h5py.File(temp_path, "r") as f:
            backend = HDF5Backend(f["data"])

            assert backend.get_shape() == shape
            assert backend.supports_batch_loading

            # Test loading
            result = backend.load_slice(n=0, c=0, z=2)
            assert result.shape == (10, 10)
            assert np.allclose(result.numpy(), data[0, 0, 2])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_tensor_like_3d_with_zarr():
    """Test TensorLike3D with ZarrBackend."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    shape = (2, 3, 5, 10, 10)
    z = zarr.zeros(shape, dtype="float32")
    data = np.random.randn(*shape).astype(np.float32)
    z[:] = data[:]

    backend = ZarrBackend(z)
    tensor_like = TensorLike3D(backend)

    assert tensor_like.shape == shape
    assert len(tensor_like) == 2

    # Test indexing
    result = tensor_like[0, 1, 2]
    assert result.shape == (10, 10)
    assert np.allclose(result.numpy(), data[0, 1, 2])


def test_backend_batch_loading():
    """Test batch loading of z-slices."""
    data = torch.randn(1, 1, 10, 20, 20)
    backend = InMemoryBackend(data)

    z_indices = [2, 3, 4, 5]
    result = backend.get_z_slices(n=0, c=0, z_indices=z_indices)

    assert result.shape == (4, 20, 20)
    for i, z in enumerate(z_indices):
        assert torch.allclose(result[i], data[0, 0, z])


if __name__ == "__main__":
    test_in_memory_backend()
    print("✓ in_memory_backend")

    test_tensor_like_3d_in_memory()
    print("✓ tensor_like_3d_in_memory")

    test_memory_mapped_backend()
    print("✓ memory_mapped_backend")

    test_zarr_backend()
    print("✓ zarr_backend")

    test_hdf5_backend()
    print("✓ hdf5_backend")

    test_tensor_like_3d_with_zarr()
    print("✓ tensor_like_3d_with_zarr")

    test_backend_batch_loading()
    print("✓ backend_batch_loading")

    print("\nAll backend tests passed!")
