"""
Tests for Laplacian pyramid functionality in stack_to_zarr.
"""

from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import zarr
except ImportError:
    zarr = None

try:
    from qlty.utils.stack_to_zarr import (
        _downsample_with_torch,
        _upsample_with_torch,
        reconstruct_from_laplacian_pyramid,
    )

    HAS_LAPLACIAN = True
except ImportError:
    HAS_LAPLACIAN = False
    _downsample_with_torch = None
    _upsample_with_torch = None
    reconstruct_from_laplacian_pyramid = None


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    yield test_dir


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.skipif(
    not HAS_LAPLACIAN, reason="Laplacian pyramid functions not available"
)
def test_downsample_with_torch_single_channel():
    """Test PyTorch downsampling for single-channel images."""
    # Create test image: (Y, X) = (64, 64)
    img = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

    # Downsample by 2x
    downsampled = _downsample_with_torch(img, y_scale=2, x_scale=2)

    assert downsampled.shape == (32, 32)
    assert downsampled.dtype == img.dtype

    # Verify it's block averaging (should be close to mean of 2x2 blocks)
    # Check a few blocks manually
    block_mean = img[0:2, 0:2].mean()
    assert abs(downsampled[0, 0] - block_mean) < 1.0  # Allow for rounding


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.skipif(
    not HAS_LAPLACIAN, reason="Laplacian pyramid functions not available"
)
def test_downsample_with_torch_multi_channel():
    """Test PyTorch downsampling for multi-channel images."""
    # Create test image: (C, Y, X) = (3, 64, 64)
    img = np.random.randint(0, 255, size=(3, 64, 64), dtype=np.uint8)

    # Downsample by 2x
    downsampled = _downsample_with_torch(img, y_scale=2, x_scale=2)

    assert downsampled.shape == (3, 32, 32)
    assert downsampled.dtype == img.dtype

    # Verify block averaging per channel
    for c in range(3):
        block_mean = img[c, 0:2, 0:2].mean()
        assert abs(downsampled[c, 0, 0] - block_mean) < 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.skipif(
    not HAS_LAPLACIAN, reason="Laplacian pyramid functions not available"
)
def test_upsample_with_torch_bilinear():
    """Test PyTorch bilinear upsampling."""
    # Create small test image: (Y, X) = (16, 16)
    img = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8).astype(np.float32)

    # Upsample to (32, 32)
    upsampled = _upsample_with_torch(img, target_size=(32, 32), mode="bilinear")

    assert upsampled.shape == (32, 32)
    assert upsampled.dtype == img.dtype

    # Bilinear upsampling should produce smooth results
    # Values should be in reasonable range
    assert upsampled.min() >= 0
    assert upsampled.max() <= 255


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.skipif(
    not HAS_LAPLACIAN, reason="Laplacian pyramid functions not available"
)
def test_upsample_with_torch_bicubic():
    """Test PyTorch bicubic upsampling."""
    # Create small test image: (Y, X) = (16, 16)
    img = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8).astype(np.float32)

    # Upsample to (32, 32)
    upsampled = _upsample_with_torch(img, target_size=(32, 32), mode="bicubic")

    assert upsampled.shape == (32, 32)
    assert upsampled.dtype == img.dtype

    # Bicubic upsampling can produce values outside [0, 255] due to interpolation
    # This is expected behavior - values should be in reasonable range
    assert upsampled.min() > -100  # Allow some negative values
    assert upsampled.max() < 400  # Allow some overshoot


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.skipif(
    not HAS_LAPLACIAN, reason="Laplacian pyramid functions not available"
)
def test_downsample_upsample_roundtrip():
    """Test that downsampling then upsampling produces reasonable results."""
    # Create test image
    img = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8).astype(np.float32)

    # Downsample
    downsampled = _downsample_with_torch(img, y_scale=2, x_scale=2)

    # Upsample back
    upsampled = _upsample_with_torch(downsampled, target_size=(64, 64), mode="bilinear")

    # Should be close but not identical (lossy operation)
    # Check that values are in reasonable range
    assert upsampled.shape == (64, 64)
    assert upsampled.min() >= 0
    assert upsampled.max() <= 255

    # Mean should be similar
    assert abs(upsampled.mean() - img.mean()) < 10.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.skipif(
    not HAS_LAPLACIAN, reason="Laplacian pyramid functions not available"
)
@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_reconstruct_from_laplacian_pyramid_single_channel(temp_dir):
    """Test reconstruction from Laplacian pyramid for single-channel images."""
    try:
        from qlty.utils.stack_to_zarr import _create_zarr_array
    except ImportError:
        pytest.skip("_create_zarr_array not available")

    # Create a simple Laplacian pyramid manually
    zarr_path = temp_dir / "test_laplacian.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    # Create base level (lowest resolution) at highest level number: (Z=1, Y=16, X=16)
    # For 2-level pyramid, base is at level "1" (following standard convention)
    base_shape = (1, 16, 16)
    base_data = np.random.randint(0, 255, size=base_shape, dtype=np.uint8)
    # Use helper function for zarr version compatibility
    _create_zarr_array(root, "1", data=base_data)

    # Create difference map level 0: (Z=1, Y=32, X=32)
    diff_shape = (1, 32, 32)
    diff_data = np.random.randint(-50, 50, size=diff_shape, dtype=np.int16).astype(
        np.float32
    )
    # Use helper function for zarr version compatibility
    _create_zarr_array(root, "diff_0", data=diff_data)

    # Reconstruct
    reconstructed = reconstruct_from_laplacian_pyramid(
        zarr_path, z_idx=0, interpolation_mode="bilinear"
    )

    assert reconstructed.shape == (32, 32)
    # Should be able to reconstruct (exact match depends on upsampling precision)
    assert reconstructed.dtype in (np.float32, np.uint8)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.skipif(
    not HAS_LAPLACIAN, reason="Laplacian pyramid functions not available"
)
@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_reconstruct_from_laplacian_pyramid_multi_channel(temp_dir):
    """Test reconstruction from Laplacian pyramid for multi-channel images."""
    try:
        from qlty.utils.stack_to_zarr import _create_zarr_array
    except ImportError:
        pytest.skip("_create_zarr_array not available")

    # Create a simple Laplacian pyramid manually
    zarr_path = temp_dir / "test_laplacian_multi.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    # Create base level (lowest resolution) at highest level number: (Z=5, C=3, Y=16, X=16) - ZCYX order
    # For 2-level pyramid, base is at level "1" (following standard convention)
    # Use Z > C to ensure correct axis order detection
    base_shape = (5, 3, 16, 16)
    base_data = np.random.randint(0, 255, size=base_shape, dtype=np.uint8)
    # Use helper function for zarr version compatibility
    _create_zarr_array(root, "1", data=base_data)

    # Create difference map level 0: (Z=5, C=3, Y=32, X=32)
    diff_shape = (5, 3, 32, 32)
    diff_data = np.random.randint(-50, 50, size=diff_shape, dtype=np.int16).astype(
        np.float32
    )
    # Use helper function for zarr version compatibility
    _create_zarr_array(root, "diff_0", data=diff_data)

    # Reconstruct
    reconstructed = reconstruct_from_laplacian_pyramid(
        zarr_path, z_idx=0, interpolation_mode="bilinear"
    )

    assert reconstructed.shape == (
        3,
        32,
        32,
    ), f"Expected (3, 32, 32), got {reconstructed.shape}"
    assert reconstructed.dtype in (np.float32, np.uint8)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.skipif(
    not HAS_LAPLACIAN, reason="Laplacian pyramid functions not available"
)
def test_laplacian_pyramid_perfect_reconstruction():
    """Test that Laplacian pyramid enables perfect reconstruction."""
    # Create a test image
    original = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8).astype(
        np.float32
    )

    # Build Gaussian pyramid (downsample progressively)
    gaussian_levels = [original]
    current = original.copy()

    # Downsample twice
    for _ in range(2):
        current = _downsample_with_torch(current, y_scale=2, x_scale=2)
        gaussian_levels.append(current)

    # Build Laplacian pyramid (difference maps)
    laplacian_levels = []
    base_level = gaussian_levels[-1]  # Lowest resolution

    # Compute differences from lowest to highest
    for i in range(len(gaussian_levels) - 1, 0, -1):
        current_level = gaussian_levels[i - 1]  # Higher resolution
        lower_level = gaussian_levels[i]  # Lower resolution

        # Upsample lower level to match current level
        upsampled = _upsample_with_torch(
            lower_level, target_size=current_level.shape[-2:], mode="bilinear"
        )

        # Compute difference
        difference = current_level - upsampled
        laplacian_levels.append((i - 1, difference))

    # Reconstruct from Laplacian pyramid
    # Start from base level (lowest resolution)
    reconstructed = base_level.copy()

    # Progressively upsample and add difference maps (from lowest to highest resolution)
    for _level_idx, diff_map in laplacian_levels:
        # Upsample current reconstruction to match difference map size
        target_size = diff_map.shape[-2:]  # (Y, X)
        reconstructed = _upsample_with_torch(
            reconstructed, target_size=target_size, mode="bilinear"
        )
        # Add difference map
        reconstructed = reconstructed + diff_map

    # Should match original (within numerical precision)
    # Note: Due to floating point precision and interpolation, exact match is not expected
    # But should be very close
    # The reconstructed should have the same shape as original
    assert (
        reconstructed.shape == original.shape
    ), f"Shape mismatch: {reconstructed.shape} vs {original.shape}"
    mse = np.mean((reconstructed - original) ** 2)
    assert mse < 1.0  # Mean squared error should be small


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.skipif(
    not HAS_LAPLACIAN, reason="Laplacian pyramid functions not available"
)
@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_laplacian_integration(temp_dir):
    """Integration test for stack_files_to_ome_zarr_laplacian."""
    try:
        from qlty.utils.stack_to_zarr import stack_files_to_ome_zarr_laplacian
    except ImportError:
        pytest.skip("stack_files_to_ome_zarr_laplacian not available")

    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(3):
        filepath = temp_dir / f"test_{i:02d}.tif"
        data = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    # Create Laplacian pyramid
    result = stack_files_to_ome_zarr_laplacian(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=3,
        interpolation_mode="bilinear",
        store_base_level=True,
        verbose=False,
    )

    assert len(result) == 1
    assert "test" in result
    metadata = result["test"]
    assert metadata["file_count"] == 3
    assert metadata["pyramid_levels"] == 3

    # Verify Zarr structure
    zarr_path = Path(metadata["zarr_path"])
    root = zarr.open_group(str(zarr_path), mode="r")

    # Should have base level at highest level number (following standard convention)
    # For 3 pyramid levels: 0, 1, 2, so base level is at "2"
    base_level_num = metadata["pyramid_levels"] - 1
    assert str(base_level_num) in root

    # Should have difference maps
    assert "diff_0" in root
    assert "diff_1" in root

    # Test reconstruction
    reconstructed = reconstruct_from_laplacian_pyramid(
        zarr_path, z_idx=0, interpolation_mode="bilinear"
    )
    assert reconstructed.shape == (64, 64)  # Full resolution
