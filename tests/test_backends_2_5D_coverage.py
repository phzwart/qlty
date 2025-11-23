"""
Additional tests for backends_2_5D to improve coverage.
Focuses on error handling, edge cases, and uncovered code paths.
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
    from_hdf5,
    from_memmap,
    from_zarr,
)


def test_in_memory_backend_invalid_shape():
    """Test error with non-5D tensor."""
    data = torch.randn(2, 3, 5, 10)  # 4D instead of 5D
    with pytest.raises(ValueError, match="must be 5D"):
        InMemoryBackend(data)


def test_in_memory_backend_load_slice_full():
    """Test loading full slice (all None)."""
    data = torch.randn(2, 3, 5, 10, 10)
    backend = InMemoryBackend(data)

    result = backend.load_slice()  # All None
    assert result.shape == data.shape


def test_in_memory_backend_load_slice_partial_indices():
    """Test loading with various partial index combinations."""
    data = torch.randn(2, 3, 5, 10, 10)
    backend = InMemoryBackend(data)

    # Test y and x slices
    result = backend.load_slice(n=0, c=0, z=2, y=slice(2, 8), x=slice(3, 7))
    assert result.shape == (6, 4)
    assert torch.allclose(result, data[0, 0, 2, 2:8, 3:7])


def test_memory_mapped_backend_invalid_shape():
    """Test error with non-5D memmap."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name

    try:
        shape = (2, 3, 5, 10)  # 4D instead of 5D
        mmap = np.memmap(temp_path, dtype="float32", mode="w+", shape=shape)
        with pytest.raises(ValueError, match="must be 5D"):
            MemoryMappedBackend(mmap)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_memory_mapped_backend_non_writable():
    """Test handling of non-writable memmap."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name

    try:
        shape = (2, 3, 5, 10, 10)
        data = np.random.randn(*shape).astype(np.float32)
        mmap = np.memmap(temp_path, dtype="float32", mode="w+", shape=shape)
        mmap[:] = data[:]
        mmap.flush()

        # Load as read-only (non-writable)
        mmap_read = np.memmap(temp_path, dtype="float32", mode="r", shape=shape)
        backend = MemoryMappedBackend(mmap_read)

        result = backend.load_slice(n=0, c=0, z=2)
        assert result.shape == (10, 10)
        assert np.allclose(result.numpy(), data[0, 0, 2])
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_memory_mapped_backend_explicit_dtype():
    """Test MemoryMappedBackend with explicit dtype conversion."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name

    try:
        shape = (2, 3, 5, 10, 10)
        mmap = np.memmap(temp_path, dtype="float64", mode="w+", shape=shape)
        mmap[:] = np.random.randn(*shape)
        mmap.flush()

        # Convert to float32
        mmap_read = np.memmap(temp_path, dtype="float64", mode="r", shape=shape)
        backend = MemoryMappedBackend(mmap_read, dtype=torch.float32)

        assert backend.get_dtype() == torch.float32
        result = backend.load_slice(n=0, c=0, z=0)
        assert result.dtype == torch.float32
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_zarr_backend_invalid_shape():
    """Test error with invalid zarr array shape."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    # 6D array - should fail (only 3D, 4D, 5D allowed)
    z = zarr.zeros((1, 1, 1, 1, 1, 1), dtype="float32")
    with pytest.raises(ValueError, match="must be 3D, 4D, or 5D"):
        ZarrBackend(z)


def test_zarr_backend_3d_shape():
    """Test ZarrBackend with 3D array (Z, Y, X)."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    z = zarr.zeros((5, 10, 10), dtype="float32")
    data = np.random.randn(5, 10, 10).astype(np.float32)
    z[:] = data[:]

    backend = ZarrBackend(z)
    assert backend.get_shape() == (1, 1, 5, 10, 10)

    result = backend.load_slice(z=2)
    assert result.shape == (10, 10)
    assert np.allclose(result.numpy(), data[2])


def test_zarr_backend_4d_shape():
    """Test ZarrBackend with 4D array (C, Z, Y, X)."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    z = zarr.zeros((3, 5, 10, 10), dtype="float32")
    data = np.random.randn(3, 5, 10, 10).astype(np.float32)
    z[:] = data[:]

    backend = ZarrBackend(z)
    assert backend.get_shape() == (1, 3, 5, 10, 10)

    result = backend.load_slice(c=1, z=2)
    assert result.shape == (10, 10)
    assert np.allclose(result.numpy(), data[1, 2])


def test_zarr_backend_explicit_dtype():
    """Test ZarrBackend with explicit dtype."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    z = zarr.zeros((2, 3, 5, 10, 10), dtype="float64")
    backend = ZarrBackend(z, dtype=torch.float32)

    assert backend.get_dtype() == torch.float32


def test_zarr_backend_load_slice_3d():
    """Test loading from 3D zarr with various slice combinations."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    z = zarr.zeros((5, 10, 10), dtype="float32")
    data = np.random.randn(5, 10, 10).astype(np.float32)
    z[:] = data[:]

    backend = ZarrBackend(z)

    # Test z slice
    result = backend.load_slice(z=slice(1, 4))
    assert result.shape == (3, 10, 10)
    assert np.allclose(result.numpy(), data[1:4])

    # Test y and x slices
    result = backend.load_slice(z=2, y=slice(2, 8), x=slice(3, 7))
    assert result.shape == (6, 4)
    assert np.allclose(result.numpy(), data[2, 2:8, 3:7])


def test_zarr_backend_load_slice_4d():
    """Test loading from 4D zarr with various slice combinations."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    z = zarr.zeros((3, 5, 10, 10), dtype="float32")
    data = np.random.randn(3, 5, 10, 10).astype(np.float32)
    z[:] = data[:]

    backend = ZarrBackend(z)

    # Test c and z slices
    result = backend.load_slice(c=1, z=slice(1, 4))
    assert result.shape == (3, 10, 10)
    assert np.allclose(result.numpy(), data[1, 1:4])


def test_zarr_backend_non_array_conversion():
    """Test zarr backend with non-ndarray conversion."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    z = zarr.zeros((5, 10, 10), dtype="float32")
    backend = ZarrBackend(z)

    # Load should still work even if zarr returns non-ndarray
    result = backend.load_slice(z=2)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 10)


def test_zarr_backend_missing_import():
    """Test error when zarr is not available."""
    # This is hard to test directly, but we can check the import error message
    try:
        import zarr  # noqa: F401
    except ImportError:
        # If zarr is not available, the error should mention installation
        pass


def test_hdf5_backend_invalid_shape():
    """Test error with invalid HDF5 dataset shape."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        with h5py.File(temp_path, "w") as f:
            dset = f.create_dataset("data", shape=(2, 3, 5), dtype="float32")  # 3D - should fail
            backend = HDF5Backend(dset)  # Should raise ValueError
    except ValueError as e:
        assert "must be 3D, 4D, or 5D" in str(e)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_hdf5_backend_3d_shape():
    """Test HDF5Backend with 3D dataset (Z, Y, X)."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        shape = (5, 10, 10)
        data = np.random.randn(*shape).astype(np.float32)

        with h5py.File(temp_path, "w") as f:
            dset = f.create_dataset("data", shape=shape, dtype="float32")
            dset[:] = data[:]

        with h5py.File(temp_path, "r") as f:
            backend = HDF5Backend(f["data"])
            assert backend.get_shape() == (1, 1, 5, 10, 10)

            result = backend.load_slice(z=2)
            assert result.shape == (10, 10)
            assert np.allclose(result.numpy(), data[2])
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_hdf5_backend_4d_shape():
    """Test HDF5Backend with 4D dataset (C, Z, Y, X)."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        shape = (3, 5, 10, 10)
        data = np.random.randn(*shape).astype(np.float32)

        with h5py.File(temp_path, "w") as f:
            dset = f.create_dataset("data", shape=shape, dtype="float32")
            dset[:] = data[:]

        with h5py.File(temp_path, "r") as f:
            backend = HDF5Backend(f["data"])
            assert backend.get_shape() == (1, 3, 5, 10, 10)

            result = backend.load_slice(c=1, z=2)
            assert result.shape == (10, 10)
            assert np.allclose(result.numpy(), data[1, 2])
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_hdf5_backend_explicit_dtype():
    """Test HDF5Backend with explicit dtype."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        with h5py.File(temp_path, "w") as f:
            dset = f.create_dataset("data", shape=(2, 3, 5, 10, 10), dtype="float64")

        with h5py.File(temp_path, "r") as f:
            backend = HDF5Backend(f["data"], dtype=torch.float32)
            assert backend.get_dtype() == torch.float32
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_hdf5_backend_load_slice_3d():
    """Test loading from 3D HDF5 with various slice combinations."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        shape = (5, 10, 10)
        data = np.random.randn(*shape).astype(np.float32)

        with h5py.File(temp_path, "w") as f:
            dset = f.create_dataset("data", shape=shape, dtype="float32")
            dset[:] = data[:]

        with h5py.File(temp_path, "r") as f:
            backend = HDF5Backend(f["data"])

            result = backend.load_slice(z=slice(1, 4))
            assert result.shape == (3, 10, 10)
            assert np.allclose(result.numpy(), data[1:4])
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_hdf5_backend_load_slice_4d():
    """Test loading from 4D HDF5 with various slice combinations."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        shape = (3, 5, 10, 10)
        data = np.random.randn(*shape).astype(np.float32)

        with h5py.File(temp_path, "w") as f:
            dset = f.create_dataset("data", shape=shape, dtype="float32")
            dset[:] = data[:]

        with h5py.File(temp_path, "r") as f:
            backend = HDF5Backend(f["data"])

            result = backend.load_slice(c=1, z=slice(1, 4))
            assert result.shape == (3, 10, 10)
            assert np.allclose(result.numpy(), data[1, 1:4])
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_tensor_like_3d_invalid_indexing():
    """Test TensorLike3D with invalid indexing type."""
    data = torch.randn(2, 3, 5, 10, 10)
    backend = InMemoryBackend(data)
    tensor_like = TensorLike3D(backend)

    with pytest.raises(TypeError, match="Unsupported indexing type"):
        tensor_like["invalid"]  # noqa: B015


def test_tensor_like_3d_slice_indexing():
    """Test TensorLike3D with slice indexing."""
    data = torch.randn(2, 3, 5, 10, 10)
    backend = InMemoryBackend(data)
    tensor_like = TensorLike3D(backend)

    result = tensor_like[0:2]
    assert result.shape == (2, 3, 5, 10, 10)
    assert torch.allclose(result, data[0:2])


def test_tensor_like_3d_tuple_indexing():
    """Test TensorLike3D with tuple indexing (all dimensions)."""
    data = torch.randn(2, 3, 5, 10, 10)
    backend = InMemoryBackend(data)
    tensor_like = TensorLike3D(backend)

    result = tensor_like[0, 1, 2, 5:8, 3:7]
    assert result.shape == (3, 4)
    assert torch.allclose(result, data[0, 1, 2, 5:8, 3:7])


def test_tensor_like_3d_repr():
    """Test TensorLike3D string representation."""
    data = torch.randn(2, 3, 5, 10, 10)
    backend = InMemoryBackend(data)
    tensor_like = TensorLike3D(backend)

    repr_str = repr(tensor_like)
    assert "TensorLike3D" in repr_str
    assert "InMemoryBackend" in repr_str


def test_backend_get_z_slices_fallback():
    """Test get_z_slices fallback for backends without batch loading."""
    # Create a mock backend without batch loading
    class NonBatchBackend(InMemoryBackend):
        @property
        def supports_batch_loading(self) -> bool:
            return False

    data = torch.randn(1, 1, 10, 20, 20)
    backend = NonBatchBackend(data)

    z_indices = [2, 3, 4, 5]
    result = backend.get_z_slices(n=0, c=0, z_indices=z_indices)

    assert result.shape == (4, 20, 20)
    for i, z in enumerate(z_indices):
        assert torch.allclose(result[i], data[0, 0, z])


def test_from_zarr():
    """Test from_zarr convenience function."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, "test.zarr")
        shape = (2, 3, 5, 10, 10)
        # Create zarr array and save data
        z = zarr.open_array(store=zarr_path, mode="w", shape=shape, dtype="float32")
        data = np.random.randn(*shape).astype(np.float32)
        z[:] = data[:]
        del z  # Close the array

        # Use from_zarr to open it
        tensor_like = from_zarr(zarr_path)
        assert tensor_like.shape == shape
        assert len(tensor_like) == 2

        result = tensor_like[0, 1, 2]
        assert result.shape == (10, 10)
        assert np.allclose(result.numpy(), data[0, 1, 2])


def test_from_hdf5():
    """Test from_hdf5 convenience function."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        shape = (2, 3, 5, 10, 10)
        data = np.random.randn(*shape).astype(np.float32)

        with h5py.File(temp_path, "w") as f:
            dset = f.create_dataset("data", shape=shape, dtype="float32")
            dset[:] = data[:]

        tensor_like = from_hdf5(temp_path, "/data")
        assert tensor_like.shape == shape
        assert len(tensor_like) == 2

        result = tensor_like[0, 1, 2]
        assert result.shape == (10, 10)
        assert np.allclose(result.numpy(), data[0, 1, 2])
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_from_memmap():
    """Test from_memmap convenience function."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name

    try:
        shape = (2, 3, 5, 10, 10)
        data = np.random.randn(*shape).astype(np.float32)
        mmap = np.memmap(temp_path, dtype="float32", mode="w+", shape=shape)
        mmap[:] = data[:]
        mmap.flush()

        tensor_like = from_memmap(temp_path, dtype=np.float32, shape=shape)
        assert tensor_like.shape == shape
        assert len(tensor_like) == 2

        result = tensor_like[0, 1, 2]
        assert result.shape == (10, 10)
        assert np.allclose(result.numpy(), data[0, 1, 2])
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

