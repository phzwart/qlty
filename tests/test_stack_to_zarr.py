"""
Tests for stack_to_zarr utility function.
"""

import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import zarr
except ImportError:
    zarr = None
    pytest.skip("zarr not available", allow_module_level=True)

from qlty.utils.stack_to_zarr import stack_files_to_zarr


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    yield test_dir
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)


def _create_test_image(filepath: Path, shape, dtype=np.uint16):
    """Create a test image file."""
    if dtype == np.uint8:
        max_val = 255
    else:
        max_val = 65535
    data = np.random.randint(0, max_val, size=shape, dtype=dtype)
    filepath = Path(filepath)

    if filepath.suffix.lower() in (".tif", ".tiff"):
        if tifffile is not None:
            tifffile.imwrite(str(filepath), data)
        elif Image is not None:
            # Convert to uint8 for PIL
            if dtype != np.uint8:
                data = (data / 256).astype(np.uint8)
            Image.fromarray(data).save(filepath)
        else:
            pytest.skip("No image library available")
    else:
        if Image is not None:
            if dtype != np.uint8:
                data = (data / 256).astype(np.uint8)
            Image.fromarray(data).save(filepath)
        else:
            pytest.skip("PIL not available")


def test_stack_files_to_zarr_single_channel(temp_dir):
    """Test stacking single channel images."""
    # Create test images
    for i in range(5):
        filepath = temp_dir / f"stack_{i:03d}.tif"
        _create_test_image(filepath, (64, 64), dtype=np.uint16)

    # Convert to zarr
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
    )

    assert len(result) == 1
    # Pattern captures basename without trailing underscore
    assert "stack" in result

    metadata = result["stack"]
    assert metadata["file_count"] == 5
    assert metadata["shape"] == (5, 64, 64)  # (Z, Y, X)
    assert metadata["axis_order"] == "ZYX"
    assert metadata["counter_range"] == (0, 4)

    # Check zarr file exists and is correct
    zarr_path = Path(metadata["zarr_path"])
    assert zarr_path.exists()

    z = zarr.open(str(zarr_path), mode="r")
    assert z.shape == (5, 64, 64)
    assert z.dtype == np.uint16

    # Check metadata
    assert z.attrs["basename"] == "stack"
    assert z.attrs["file_count"] == 5
    assert z.attrs["counter_range"] == [0, 4]
    assert z.attrs["axis_order"] == "ZYX"


def test_stack_files_to_zarr_multi_channel(temp_dir):
    """Test stacking multi-channel images."""
    # Create test images with 3 channels
    for i in range(4):
        filepath = temp_dir / f"image_{i:03d}.tif"
        _create_test_image(filepath, (3, 32, 32), dtype=np.uint16)

    # Convert to zarr with default axis order
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="ZCYX",
    )

    assert len(result) == 1
    metadata = result["image"]
    assert metadata["file_count"] == 4
    assert metadata["shape"] == (4, 3, 32, 32)  # (Z, C, Y, X)
    assert metadata["axis_order"] == "ZCYX"

    # Check zarr file
    zarr_path = Path(metadata["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.shape == (4, 3, 32, 32)


def test_stack_files_to_zarr_axis_order_czyx(temp_dir):
    """Test different axis order (CZYX)."""
    # Create test images with 2 channels
    for i in range(3):
        filepath = temp_dir / f"data_{i:02d}.tif"
        _create_test_image(filepath, (2, 16, 16), dtype=np.uint8)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="CZYX",
    )

    metadata = result["data"]
    assert metadata["shape"] == (2, 3, 16, 16)  # (C, Z, Y, X)
    assert metadata["axis_order"] == "CZYX"

    zarr_path = Path(metadata["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.shape == (2, 3, 16, 16)


def test_stack_files_to_zarr_multiple_stacks(temp_dir):
    """Test multiple stacks in same directory."""
    # Create two different stacks
    for i in range(3):
        _create_test_image(temp_dir / f"stack1_{i:03d}.tif", (20, 20))
        _create_test_image(temp_dir / f"stack2_{i:03d}.tif", (30, 30))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
    )

    assert len(result) == 2
    assert "stack1" in result
    assert "stack2" in result

    assert result["stack1"]["shape"] == (3, 20, 20)
    assert result["stack2"]["shape"] == (3, 30, 30)


def test_stack_files_to_zarr_custom_output_dir(temp_dir):
    """Test custom output directory."""
    output_dir = temp_dir / "zarr_output"
    output_dir.mkdir()

    for i in range(3):
        _create_test_image(temp_dir / f"test_{i:02d}.tif", (10, 10))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        output_dir=output_dir,
    )

    metadata = result["test"]
    zarr_path = Path(metadata["zarr_path"])
    assert zarr_path.parent == output_dir
    assert zarr_path.name == "test.zarr"


def test_stack_files_to_zarr_custom_naming(temp_dir):
    """Test custom output naming function."""
    for i in range(2):
        _create_test_image(temp_dir / f"original_{i:01d}.tif", (8, 8))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        output_naming=lambda basename: f"{basename}processed.zarr",
    )

    metadata = result["original"]
    zarr_path = Path(metadata["zarr_path"])
    assert zarr_path.name == "originalprocessed.zarr"


def test_stack_files_to_zarr_dry_run(temp_dir):
    """Test dry run mode (no zarr creation)."""
    for i in range(3):
        _create_test_image(temp_dir / f"dry_{i:01d}.tif", (12, 12))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        dry_run=True,
    )

    assert len(result) == 1
    metadata = result["dry"]
    assert metadata["file_count"] == 3
    assert metadata["shape"] == (3, 12, 12)

    # Check zarr file was NOT created
    zarr_path = Path(metadata["zarr_path"])
    assert not zarr_path.exists()


def test_stack_files_to_zarr_gaps_in_sequence(temp_dir):
    """Test handling gaps in counter sequence."""
    # Create files with gaps: 0, 1, 3, 5
    for i in [0, 1, 3, 5]:
        _create_test_image(temp_dir / f"gap_{i:01d}.tif", (10, 10))

    # Should work but warn about gaps
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
    )

    metadata = result["gap"]
    assert metadata["file_count"] == 4
    assert metadata["counter_range"] == (0, 5)

    # Zarr should have 4 slices (not 6)
    zarr_path = Path(metadata["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.shape[0] == 4  # Z dimension


def test_stack_files_to_zarr_custom_dtype(temp_dir):
    """Test custom dtype conversion."""
    for i in range(2):
        _create_test_image(temp_dir / f"dtype_{i:01d}.tif", (5, 5), dtype=np.uint16)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        dtype=np.float32,
    )

    zarr_path = Path(result["dtype"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.dtype == np.float32


def test_stack_files_to_zarr_custom_chunks(temp_dir):
    """Test custom chunk size."""
    for i in range(3):
        _create_test_image(temp_dir / f"chunk_{i:01d}.tif", (20, 20))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        zarr_chunks=(1, 10, 10),
    )

    zarr_path = Path(result["chunk"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.chunks == (1, 10, 10)


def test_stack_files_to_zarr_no_sort(temp_dir):
    """Test without sorting by counter."""
    # Create files in non-sequential order
    for i in [3, 1, 4, 0, 2]:
        _create_test_image(temp_dir / f"nosort_{i:01d}.tif", (6, 6))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        sort_by_counter=False,
    )

    metadata = result["nosort"]
    # Files should be in original order (not sorted)
    # But zarr should still be created correctly
    assert metadata["file_count"] == 5


def test_stack_files_to_zarr_different_pattern(temp_dir):
    """Test different pattern matching."""
    # Files with different pattern: name_z001.tif
    for i in range(3):
        _create_test_image(temp_dir / f"data_z{i:03d}.tif", (8, 8))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_z(\d+)\.tif$",
    )

    assert "data" in result
    metadata = result["data"]
    assert metadata["file_count"] == 3


def test_stack_files_to_zarr_dimension_mismatch_error(temp_dir):
    """Test error when images have different dimensions."""
    # Create images with different sizes
    _create_test_image(temp_dir / "mismatch_0.tif", (10, 10))
    _create_test_image(temp_dir / "mismatch_1.tif", (20, 20))  # Different size!

    with pytest.raises(ValueError, match="has shape"):
        stack_files_to_zarr(
            directory=temp_dir,
            extension=".tif",
            pattern=r"(.+)_(\d+)\.tif$",
        )


def test_stack_files_to_zarr_invalid_directory():
    """Test error with invalid directory."""
    with pytest.raises(ValueError, match="Directory does not exist"):
        stack_files_to_zarr(
            directory="/nonexistent/path",
            extension=".tif",
            pattern=r"(.+)_(\d+)\.tif$",
        )


def test_stack_files_to_zarr_no_matching_files(temp_dir):
    """Test with no matching files."""
    # Create file that doesn't match pattern
    _create_test_image(temp_dir / "nomatch.tif", (5, 5))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
    )

    assert len(result) == 0


def test_stack_files_to_zarr_wrong_extension(temp_dir):
    """Test files with wrong extension are ignored."""
    # Create .png files but look for .tif
    for i in range(2):
        _create_test_image(temp_dir / f"test_{i:01d}.png", (5, 5))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
    )

    assert len(result) == 0


def test_stack_files_to_zarr_png_files(temp_dir):
    """Test with PNG files (using PIL fallback)."""
    if Image is None:
        pytest.skip("PIL not available")

    for i in range(2):
        _create_test_image(temp_dir / f"png_{i:01d}.png", (6, 6), dtype=np.uint8)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".png",
        pattern=r"(.+)_(\d+)\.png$",
    )

    assert len(result) == 1
    assert result["png"]["file_count"] == 2


def test_stack_files_to_zarr_invalid_axis_order(temp_dir):
    """Test error with invalid axis order."""
    for i in range(2):
        _create_test_image(temp_dir / f"test_{i:01d}.tif", (2, 5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="axis_order must contain"):
        stack_files_to_zarr(
            directory=temp_dir,
            extension=".tif",
            pattern=r"(.+)_(\d+)\.tif$",
            axis_order="ZYX",  # Missing C for multi-channel
        )


def test_stack_files_to_zarr_single_channel_ignores_axis_order(temp_dir):
    """Test that single channel always uses ZYX regardless of axis_order."""
    for i in range(2):
        _create_test_image(temp_dir / f"single_{i:01d}.tif", (5, 5))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="CZYX",  # Should be ignored for single channel
    )

    metadata = result["single"]
    assert metadata["axis_order"] == "ZYX"
    assert metadata["shape"] == (2, 5, 5)


def test_stack_files_to_zarr_metadata_storage(temp_dir):
    """Test that metadata is stored in zarr attributes."""
    for i in range(3):
        _create_test_image(temp_dir / f"meta_{i:01d}.tif", (7, 7))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
    )

    zarr_path = Path(result["meta"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")

    # Check all metadata is stored
    assert "basename" in z.attrs
    assert "file_count" in z.attrs
    assert "counter_range" in z.attrs
    assert "axis_order" in z.attrs
    assert "files" in z.attrs
    assert "pattern" in z.attrs
    assert "extension" in z.attrs

    # Check values
    assert z.attrs["basename"] == "meta"
    assert z.attrs["file_count"] == 3
    assert len(z.attrs["files"]) == 3


def test_stack_files_to_zarr_zarr_data_correctness(temp_dir):
    """Test that zarr data matches original images."""
    # Create images with known values
    for i in range(3):
        filepath = temp_dir / f"check_{i:01d}.tif"
        # Create image with value = i in all pixels
        data = np.full((10, 10), i, dtype=np.uint16)
        if tifffile is not None:
            tifffile.imwrite(str(filepath), data)
        else:
            pytest.skip("tifffile not available")

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
    )

    zarr_path = Path(result["check"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")

    # Check each z-slice
    for i in range(3):
        slice_data = z[i]
        assert np.all(slice_data == i)


def test_stack_files_to_zarr_pattern_with_one_group_error(temp_dir):
    """Test error when pattern has only one group."""
    _create_test_image(temp_dir / "test_0.tif", (5, 5))

    with pytest.raises(ValueError, match="Pattern must have at least 2 groups"):
        stack_files_to_zarr(
            directory=temp_dir,
            extension=".tif",
            pattern=r"test_\d+\.tif$",  # Only one group
        )


def test_stack_files_to_zarr_unsupported_image_dimensions(temp_dir):
    """Test error with unsupported image dimensions."""
    # Create 1D image (unsupported)
    data = np.random.randint(0, 255, size=(100,), dtype=np.uint8)
    filepath = temp_dir / "unsupported_0.tif"
    if tifffile is not None:
        tifffile.imwrite(str(filepath), data)
    else:
        pytest.skip("tifffile not available")

    with pytest.raises(ValueError, match="Unsupported image dimensions"):
        stack_files_to_zarr(
            directory=temp_dir,
            extension=".tif",
            pattern=r"(.+)_(\d+)\.tif$",
        )


def test_stack_files_to_zarr_extension_normalization(temp_dir):
    """Test that extension is normalized (with/without dot)."""
    for i in range(2):
        _create_test_image(temp_dir / f"ext_{i:01d}.tif", (5, 5))

    # Test with extension without dot
    result1 = stack_files_to_zarr(
        directory=temp_dir,
        extension="tif",  # No dot
        pattern=r"(.+)_(\d+)\.tif$",
    )

    # Test with extension with dot
    result2 = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",  # With dot
        pattern=r"(.+)_(\d+)\.tif$",
    )

    assert len(result1) == len(result2) == 1


def test_stack_files_to_zarr_case_insensitive_extension(temp_dir):
    """Test that extension matching is case insensitive."""
    for i in range(2):
        _create_test_image(temp_dir / f"case_{i:01d}.TIF", (5, 5))  # Uppercase

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",  # Lowercase
        pattern=r"(.+)_(\d+)\.TIF$",
    )

    assert len(result) == 1
