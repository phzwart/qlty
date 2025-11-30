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

    test_tensor_like_3d_in_memory()

    test_memory_mapped_backend()

    test_zarr_backend()

    test_hdf5_backend()

    test_tensor_like_3d_with_zarr()

    test_backend_batch_loading()


# ============================================================================
# Additional coverage tests for error handling and edge cases
# ============================================================================


def test_memory_mapped_backend_with_y_x():
    """Test MemoryMappedBackend with y and x parameters."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name

    try:
        shape = (2, 3, 5, 10, 10)
        data = np.random.randn(*shape).astype(np.float32)
        mmap = np.memmap(temp_path, dtype="float32", mode="w+", shape=shape)
        mmap[:] = data[:]
        mmap.flush()

        mmap_read = np.memmap(temp_path, dtype="float32", mode="r", shape=shape)
        backend = MemoryMappedBackend(mmap_read)

        # Test with y slice and x parameter (avoid scalar case which may have issues)
        result = backend.load_slice(n=0, c=0, z=2, y=slice(2, 5), x=3)
        assert result.shape == (3,)  # 1D tensor
        assert np.allclose(result.numpy(), data[0, 0, 2, 2:5, 3])

        # Test with y parameter and x slice
        result = backend.load_slice(n=0, c=0, z=2, y=5, x=slice(2, 5))
        assert result.shape == (3,)  # 1D tensor
        assert np.allclose(result.numpy(), data[0, 0, 2, 5, 2:5])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_zarr_backend_import_error(monkeypatch):
    """Test ZarrBackend raises ImportError when zarr is not available."""
    try:
        import zarr

        zarr_available = True
    except ImportError:
        zarr_available = False

    if not zarr_available:
        pytest.skip("zarr not available, cannot test import error")

    # Create a mock zarr array first
    zarr_array = zarr.zeros((2, 3, 5, 10, 10), dtype="float32")

    # Mock the import to fail
    def mock_import(name, *args, **kwargs):
        if name == "zarr" or name.startswith("zarr."):
            msg = "zarr not available"
            raise ImportError(msg)
        # Use original import for everything else
        import builtins

        return builtins.__import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    # Should raise ImportError
    with pytest.raises(ImportError, match="zarr is required"):
        ZarrBackend(zarr_array)


def test_zarr_backend_with_y_x():
    """Test ZarrBackend with y and x parameters."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    # Create zarr array
    zarr_array = zarr.zeros((2, 3, 5, 10, 10), dtype="float32")
    data = np.random.randn(2, 3, 5, 10, 10).astype(np.float32)
    zarr_array[:] = data

    backend = ZarrBackend(zarr_array)

    # Test with y slice and x parameter (avoid scalar case)
    result = backend.load_slice(n=0, c=1, z=2, y=slice(4, 6), x=3)
    assert result.shape == (2,)  # 1D tensor
    assert np.allclose(result.numpy(), data[0, 1, 2, 4:6, 3])


def test_zarr_backend_3d_shape():
    """Test ZarrBackend with 3D array (Z, Y, X)."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    zarr_array = zarr.zeros((5, 10, 10), dtype="float32")
    data = np.random.randn(5, 10, 10).astype(np.float32)
    zarr_array[:] = data

    backend = ZarrBackend(zarr_array)

    # Should handle 3D shape
    assert backend.shape == (1, 1, 5, 10, 10)

    # Test loading - when n is int, returns (C, Z, Y, X) = (1, Z, Y, X)
    # But when z is int, it becomes (1, 1, Y, X)
    result = backend.load_slice(n=0, c=0, z=2)
    assert result.shape == (1, 1, 10, 10)  # Normalized shape
    assert np.allclose(result.numpy()[0, 0], data[2])


def test_zarr_backend_4d_shape():
    """Test ZarrBackend with 4D array (C, Z, Y, X)."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    zarr_array = zarr.zeros((3, 5, 10, 10), dtype="float32")
    data = np.random.randn(3, 5, 10, 10).astype(np.float32)
    zarr_array[:] = data

    backend = ZarrBackend(zarr_array)

    # Should handle 4D shape
    assert backend.shape == (1, 3, 5, 10, 10)

    # Test loading - when n is int, returns (C, Z, Y, X)
    # But when z is int, it becomes (1, 1, Y, X)
    result = backend.load_slice(n=0, c=1, z=2)
    assert result.shape == (1, 1, 10, 10)  # Normalized shape
    assert np.allclose(result.numpy()[0, 0], data[1, 2])


def test_hdf5_backend_import_error(monkeypatch):
    """Test HDF5Backend raises ImportError when h5py is not available."""
    try:
        import h5py

        h5py_available = True
    except ImportError:
        h5py_available = False

    if not h5py_available:
        pytest.skip("h5py not available, cannot test import error")

    # Create a mock h5 dataset
    class MockH5Dataset:
        def __init__(self):
            self.shape = (2, 3, 5, 10, 10)
            self.dtype = np.dtype("float32")

    mock_dataset = MockH5Dataset()

    # Mock the import to fail
    def mock_import(name, *args, **kwargs):
        if name == "h5py" or name.startswith("h5py."):
            msg = "h5py not available"
            raise ImportError(msg)
        # Use original import for everything else
        import builtins

        return builtins.__import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    # Should raise ImportError
    with pytest.raises(ImportError, match="h5py is required"):
        HDF5Backend(mock_dataset)


def test_hdf5_backend_invalid_shape():
    """Test HDF5Backend with invalid shape."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    # Create mock dataset with invalid shape (2D)
    class MockH5Dataset:
        def __init__(self, shape):
            self.shape = shape
            self.dtype = np.dtype("float32")

    # Invalid: 2D shape
    with pytest.raises(ValueError, match="HDF5 dataset must be 3D, 4D, or 5D"):
        HDF5Backend(MockH5Dataset((10, 10)))

    # Invalid: 6D shape
    with pytest.raises(ValueError, match="HDF5 dataset must be 3D, 4D, or 5D"):
        HDF5Backend(MockH5Dataset((1, 1, 1, 1, 1, 1)))


def test_hdf5_backend_3d_shape():
    """Test HDF5Backend with 3D array (Z, Y, X)."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        with h5py.File(temp_path, "w") as f:
            dataset = f.create_dataset("data", shape=(5, 10, 10), dtype="float32")
            data = np.random.randn(5, 10, 10).astype(np.float32)
            dataset[:] = data

        with h5py.File(temp_path, "r") as f:
            backend = HDF5Backend(f["data"])

            # Should handle 3D shape
            assert backend.shape == (1, 1, 5, 10, 10)

            # Test loading
            result = backend.load_slice(n=0, c=0, z=2)
            assert result.shape == (10, 10)
            assert np.allclose(result.numpy(), data[2])
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_hdf5_backend_4d_shape():
    """Test HDF5Backend with 4D array (C, Z, Y, X)."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        with h5py.File(temp_path, "w") as f:
            dataset = f.create_dataset("data", shape=(3, 5, 10, 10), dtype="float32")
            data = np.random.randn(3, 5, 10, 10).astype(np.float32)
            dataset[:] = data

        with h5py.File(temp_path, "r") as f:
            backend = HDF5Backend(f["data"])

            # Should handle 4D shape
            assert backend.shape == (1, 3, 5, 10, 10)

            # Test loading
            result = backend.load_slice(n=0, c=1, z=2)
            assert result.shape == (10, 10)
            assert np.allclose(result.numpy(), data[1, 2])
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
