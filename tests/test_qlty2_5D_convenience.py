"""
Tests for 2.5D Quilt convenience methods and extract_patch_pairs integration.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from qlty.backends_2_5D import from_hdf5, from_zarr
from qlty.qlty2_5D import NCZYX25DQuilt


def test_extract_patch_pairs():
    """Test extract_patch_pairs integration."""
    data = torch.randn(2, 1, 10, 50, 50)

    spec = {"identity": [0]}
    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[5],
    )

    patches1, patches2, deltas, rotations = quilt.extract_patch_pairs(
        window=(16, 16),
        num_patches=10,
        delta_range=(4.0, 8.0),
        random_seed=42,
    )

    # Check shapes
    assert patches1.shape == (20, 1, 16, 16)  # 2 images * 10 patches
    assert patches2.shape == (20, 1, 16, 16)
    assert deltas.shape == (20, 2)
    assert rotations.shape == (20,)


def test_extract_patch_pairs_requires_2d_mode():
    """Test that extract_patch_pairs requires 2d mode."""
    data = torch.randn(1, 1, 10, 50, 50)

    spec = {"identity": [0]}
    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="3d",  # Wrong mode
    )

    with pytest.raises(ValueError, match="accumulation_mode='2d'"):
        quilt.extract_patch_pairs(
            window=(16, 16),
            num_patches=10,
            delta_range=(4.0, 8.0),
        )


def test_to_ncyx_quilt():
    """Test to_ncyx_quilt convenience method."""
    data = torch.randn(1, 1, 10, 100, 100)

    spec = {"identity": [0]}
    quilt_2_5d = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[5],
    )

    quilt_2d = quilt_2_5d.to_ncyx_quilt(window=(32, 32), step=(16, 16), border=(4, 4))

    # Check that it's a valid 2D quilt
    assert quilt_2d.Y == 100
    assert quilt_2d.X == 100
    assert quilt_2d.window == (32, 32)
    assert quilt_2d.step == (16, 16)


def test_to_ncyx_quilt_requires_2d_mode():
    """Test that to_ncyx_quilt requires 2d mode."""
    data = torch.randn(1, 1, 10, 100, 100)

    spec = {"identity": [0]}
    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="3d",  # Wrong mode
    )

    with pytest.raises(ValueError, match="accumulation_mode='2d'"):
        quilt.to_ncyx_quilt(window=(32, 32), step=(16, 16))


def test_from_zarr():
    """Test from_zarr convenience function."""
    try:
        import zarr
    except ImportError:
        pytest.skip("zarr not available")

    # Create zarr array
    shape = (1, 1, 10, 20, 20)
    z = zarr.zeros(shape, dtype="float32")
    data = np.random.randn(*shape).astype(np.float32)
    z[:] = data[:]

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, "test.zarr")
        # Create zarr array directly in directory
        z_saved = zarr.open_array(
            store=zarr_path,
            mode="w",
            shape=shape,
            dtype="float32",
        )
        z_saved[:] = data[:]

        # Load using convenience function
        data_tensor = from_zarr(zarr_path)
        quilt = NCZYX25DQuilt(
            data_source=data_tensor,
            channel_spec={"identity": [0]},
            accumulation_mode="2d",
            z_slices=[5],
        )

        result = quilt.convert()
        assert result.shape == (1, 1, 20, 20)


def test_from_hdf5():
    """Test from_hdf5 convenience function."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    shape = (1, 1, 10, 20, 20)
    data = np.random.randn(*shape).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    try:
        with h5py.File(temp_path, "w") as f:
            dset = f.create_dataset("data", shape=shape, dtype="float32")
            dset[:] = data[:]

        # Load using convenience function
        data_tensor = from_hdf5(temp_path, dataset_path="data")
        quilt = NCZYX25DQuilt(
            data_source=data_tensor,
            channel_spec={"identity": [0]},
            accumulation_mode="2d",
            z_slices=[5],
        )

        result = quilt.convert()
        assert result.shape == (1, 1, 20, 20)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    test_extract_patch_pairs()

    test_extract_patch_pairs_requires_2d_mode()

    test_to_ncyx_quilt()

    test_to_ncyx_quilt_requires_2d_mode()

    test_from_zarr()

    test_from_hdf5()
