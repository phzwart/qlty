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

import pytest

try:
    import zarr
except ImportError:
    zarr = None
    pytest.skip("zarr not available", allow_module_level=True)

# Import normally - using coverage run directly (not pytest-cov) avoids torch import conflicts
# during test collection, so we can use normal imports for proper coverage tracking
from qlty.utils.stack_to_zarr import stack_files_to_ome_zarr, stack_files_to_zarr


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


def test_stack_files_to_zarr_pil_fallback(temp_dir, monkeypatch):
    """Test PIL fallback when tifffile is not available."""
    if Image is None:
        pytest.skip("PIL not available")

    # Mock tifffile to be None to force PIL fallback
    import qlty.utils.stack_to_zarr as stack_module

    original_tifffile = stack_module.tifffile
    stack_module.tifffile = None

    try:
        for i in range(2):
            _create_test_image(temp_dir / f"pil_{i:01d}.tif", (5, 5), dtype=np.uint8)

        result = stack_files_to_zarr(
            directory=temp_dir,
            extension=".tif",
            pattern=r"(.+)_(\d+)\.tif$",
        )

        assert len(result) == 1
        assert result["pil"]["file_count"] == 2
    finally:
        stack_module.tifffile = original_tifffile


def test_stack_files_to_zarr_no_image_library_error(temp_dir, monkeypatch):
    """Test error when no image library is available."""
    import qlty.utils.stack_to_zarr as stack_module

    original_tifffile = stack_module.tifffile
    original_image = stack_module.Image
    stack_module.tifffile = None
    stack_module.Image = None

    try:
        # Create a file that would need to be loaded
        filepath = temp_dir / "test_0.tif"
        filepath.touch()  # Create empty file

        with pytest.raises(
            RuntimeError, match="Cannot load image.*No suitable library"
        ):
            stack_files_to_zarr(
                directory=temp_dir,
                extension=".tif",
                pattern=r"(.+)_(\d+)\.tif$",
            )
    finally:
        stack_module.tifffile = original_tifffile
        stack_module.Image = original_image


def test_stack_files_to_zarr_yxc_format(temp_dir):
    """Test images in (Y, X, C) format (last dim <= 4)."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create RGB images (Y, X, C) format
    for i in range(3):
        filepath = temp_dir / f"rgb_{i:01d}.tif"
        # Create (Y, X, C) image
        data = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="ZCYX",
    )

    assert len(result) == 1
    metadata = result["rgb"]
    assert metadata["shape"] == (3, 3, 10, 10)  # (Z, C, Y, X)
    assert metadata["axis_order"] == "ZCYX"


def test_stack_files_to_zarr_axis_reordering_czyx(temp_dir):
    """Test axis reordering from ZCYX to CZYX."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create multi-channel images
    for i in range(2):
        filepath = temp_dir / f"reorder_{i:01d}.tif"
        data = np.random.randint(0, 255, size=(2, 8, 8), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="CZYX",
    )

    zarr_path = Path(result["reorder"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.shape == (2, 2, 8, 8)  # (C, Z, Y, X)
    assert result["reorder"]["axis_order"] == "CZYX"


def test_stack_files_to_zarr_custom_chunks_multi_channel(temp_dir):
    """Test custom chunks for multi-channel images."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    for i in range(3):
        filepath = temp_dir / f"chunkmc_{i:01d}.tif"
        data = np.random.randint(0, 255, size=(3, 16, 16), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="ZCYX",
        zarr_chunks=(1, 1, 8, 8),
    )

    zarr_path = Path(result["chunkmc"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.chunks == (1, 1, 8, 8)


def test_stack_files_to_zarr_dtype_conversion_multi_channel(temp_dir):
    """Test dtype conversion for multi-channel images."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    for i in range(2):
        filepath = temp_dir / f"dtypemc_{i:01d}.tif"
        data = np.random.randint(0, 255, size=(2, 6, 6), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        dtype=np.float32,
    )

    zarr_path = Path(result["dtypemc"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.dtype == np.float32


def test_stack_files_to_zarr_data_correctness_multi_channel(temp_dir):
    """Test data correctness for multi-channel images."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create images with known values
    for i in range(2):
        filepath = temp_dir / f"checkmc_{i:01d}.tif"
        # Create image with channel value = i+1, pixel value = channel
        data = np.full((2, 5, 5), i + 1, dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="ZCYX",
    )

    zarr_path = Path(result["checkmc"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")

    # Check first z-slice, first channel should be all 1s
    assert np.all(z[0, 0] == 1)
    # Check second z-slice, first channel should be all 2s
    assert np.all(z[1, 0] == 2)


def test_stack_files_to_zarr_pattern_compiled_regex(temp_dir):
    """Test with pre-compiled regex pattern."""
    for i in range(2):
        _create_test_image(temp_dir / f"compiled_{i:01d}.tif", (5, 5))

    pattern = re.compile(r"(.+)_(\d+)\.tif$")
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=pattern,
    )

    assert len(result) == 1
    assert result["compiled"]["file_count"] == 2


def test_stack_files_to_zarr_counter_not_parseable(temp_dir):
    """Test handling of non-parseable counter values."""
    # Create file with non-numeric counter
    _create_test_image(temp_dir / "test_abc.tif", (5, 5))
    # Create file with valid counter
    _create_test_image(temp_dir / "test_0.tif", (5, 5))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\w+)\.tif$",  # \w+ matches both abc and 0
    )

    # Should only process the one with parseable counter
    assert len(result) == 1
    assert result["test"]["file_count"] == 1


def test_stack_files_to_zarr_multi_channel_shape_validation(temp_dir):
    """Test shape validation for multi-channel images."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create images with different channel counts
    data1 = np.random.randint(0, 255, size=(2, 10, 10), dtype=np.uint8)
    tifffile.imwrite(str(temp_dir / "shape_0.tif"), data1)
    data2 = np.random.randint(0, 255, size=(3, 10, 10), dtype=np.uint8)  # Different!
    tifffile.imwrite(str(temp_dir / "shape_1.tif"), data2)

    with pytest.raises(ValueError, match="has shape"):
        stack_files_to_zarr(
            directory=temp_dir,
            extension=".tif",
            pattern=r"(.+)_(\d+)\.tif$",
        )


def test_stack_files_to_zarr_yxc_shape_validation(temp_dir):
    """Test shape validation for (Y, X, C) format images."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create (Y, X, C) images with different sizes
    data1 = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
    tifffile.imwrite(str(temp_dir / "yxc_0.tif"), data1)
    data2 = np.random.randint(0, 255, size=(15, 15, 3), dtype=np.uint8)  # Different!
    tifffile.imwrite(str(temp_dir / "yxc_1.tif"), data2)

    with pytest.raises(ValueError, match="has shape"):
        stack_files_to_zarr(
            directory=temp_dir,
            extension=".tif",
            pattern=r"(.+)_(\d+)\.tif$",
        )


def test_stack_files_to_zarr_default_chunks_zcyx(temp_dir):
    """Test default chunk calculation for ZCYX order."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    for i in range(2):
        filepath = temp_dir / f"chunkzcyx_{i:01d}.tif"
        data = np.random.randint(0, 255, size=(3, 12, 12), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="ZCYX",
    )

    zarr_path = Path(result["chunkzcyx"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    # Default should be (1, C, Y, X)
    assert z.chunks[0] == 1
    assert z.chunks[1] == 3  # C


def test_stack_files_to_zarr_default_chunks_czyx(temp_dir):
    """Test default chunk calculation for CZYX order."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    for i in range(2):
        filepath = temp_dir / f"chunkczyx_{i:01d}.tif"
        data = np.random.randint(0, 255, size=(3, 12, 12), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="CZYX",
    )

    zarr_path = Path(result["chunkczyx"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    # Default should be (C, 1, Y, X)
    assert z.chunks[0] == 3  # C
    assert z.chunks[1] == 1


def test_stack_files_to_zarr_metadata_pattern_string(temp_dir):
    """Test that pattern is stored correctly as string in metadata."""
    for i in range(2):
        _create_test_image(temp_dir / f"pattern_{i:01d}.tif", (5, 5))

    pattern_str = r"(.+)_(\d+)\.tif$"
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=pattern_str,
    )

    zarr_path = Path(result["pattern"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.attrs["pattern"] == pattern_str


def test_stack_files_to_zarr_metadata_pattern_compiled(temp_dir):
    """Test that compiled pattern is stored correctly in metadata."""
    for i in range(2):
        _create_test_image(temp_dir / f"patternc_{i:01d}.tif", (5, 5))

    pattern = re.compile(r"(.+)_(\d+)\.tif$")
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=pattern,
    )

    zarr_path = Path(result["patternc"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.attrs["pattern"] == pattern.pattern


def test_stack_files_to_zarr_subdirectories_ignored(temp_dir):
    """Test that subdirectories are ignored."""
    # Create a subdirectory
    subdir = temp_dir / "subdir"
    subdir.mkdir()

    # Create images in both main dir and subdir
    for i in range(2):
        _create_test_image(temp_dir / f"main_{i:01d}.tif", (5, 5))
        _create_test_image(subdir / f"sub_{i:01d}.tif", (5, 5))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
    )

    # Should only find files in main directory, not subdirectory
    assert len(result) == 1
    assert "main" in result
    assert result["main"]["file_count"] == 2


def test_stack_files_to_zarr_generic_axis_order_chunks(temp_dir):
    """Test default chunk calculation for generic axis orders (not ZCYX or CZYX)."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Test with ZYCX axis order
    for i in range(2):
        filepath = temp_dir / f"generic_{i:01d}.tif"
        data = np.random.randint(0, 255, size=(2, 8, 8), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="ZYCX",  # Different from ZCYX or CZYX
    )

    zarr_path = Path(result["generic"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    # Generic chunks: first dimension as 1
    assert z.chunks[0] == 1


def test_stack_files_to_zarr_pattern_no_groups(temp_dir):
    """Test error when pattern has no groups but matches."""
    _create_test_image(temp_dir / "test_0.tif", (5, 5))

    # Pattern with no groups but matches the filename
    with pytest.raises(ValueError, match="Pattern has no groups"):
        stack_files_to_zarr(
            directory=temp_dir,
            extension=".tif",
            pattern=r"test_\d+\.tif$",  # Matches but no groups
        )


def test_stack_files_to_zarr_pattern_no_match_but_has_groups(temp_dir):
    """Test that pattern matching properly skips non-matching files."""
    # Create files that match and don't match
    _create_test_image(temp_dir / "match_0.tif", (5, 5))
    _create_test_image(temp_dir / "nomatch_file.tif", (5, 5))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",  # Requires counter
    )

    # Should only match the file with counter
    assert len(result) == 1
    assert "match" in result


def test_stack_files_to_zarr_extension_normalization_no_dot(temp_dir):
    """Test extension normalization when extension doesn't start with dot."""
    for i in range(2):
        _create_test_image(temp_dir / f"test_{i:01d}.tif", (5, 5))

    # Test with extension without leading dot
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension="tif",  # No dot
        pattern=r"(.+)_(\d+)\.tif$",
    )

    assert len(result) == 1
    assert result["test"]["file_count"] == 2


def test_stack_files_to_zarr_gaps_warning(capsys, temp_dir):
    """Test that gaps in sequence produce warning output."""
    # Create files with gaps: 0, 2, 5
    for i in [0, 2, 5]:
        _create_test_image(temp_dir / f"gap_{i:01d}.tif", (10, 10))

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
    )

    # Check that warning was printed
    captured = capsys.readouterr()
    assert "missing counters" in captured.out.lower()
    assert "gap" in captured.out.lower()

    assert len(result) == 1
    metadata = result["gap"]
    assert metadata["file_count"] == 3
    assert metadata["counter_range"] == (0, 5)


def test_stack_files_to_zarr_axis_order_same_zcyx(temp_dir):
    """Test that same axis order (ZCYX) works correctly."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create multi-channel images
    for i in range(2):
        filepath = temp_dir / f"same_{i:01d}.tif"
        data = np.random.randint(0, 255, size=(3, 8, 8), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="ZCYX",  # Same as default
    )

    metadata = result["same"]
    assert metadata["axis_order"] == "ZCYX"
    assert metadata["shape"] == (2, 3, 8, 8)


def test_stack_files_to_zarr_dtype_conversion_required(temp_dir):
    """Test dtype conversion when explicitly needed."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create uint16 images but convert to float32
    for i in range(2):
        filepath = temp_dir / f"convert_{i:01d}.tif"
        data = np.random.randint(0, 65535, size=(10, 10), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        dtype=np.float32,
    )

    zarr_path = Path(result["convert"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.dtype == np.float32
    # Check that data was actually converted
    assert z[0].dtype == np.float32


def test_stack_files_to_zarr_multi_channel_dtype_conversion(temp_dir):
    """Test dtype conversion for multi-channel images during stacking."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create uint8 images but convert to uint16
    for i in range(2):
        filepath = temp_dir / f"mcconvert_{i:01d}.tif"
        data = np.random.randint(0, 255, size=(2, 6, 6), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        dtype=np.uint16,
    )

    zarr_path = Path(result["mcconvert"]["zarr_path"])
    z = zarr.open(str(zarr_path), mode="r")
    assert z.dtype == np.uint16


def test_stack_files_to_zarr_empty_directory(temp_dir):
    """Test with empty directory."""
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
    )

    assert len(result) == 0
    assert result == {}


def test_stack_files_to_zarr_multiprocessing(temp_dir):
    """Test multiprocessing support for image loading."""
    # Create multiple images
    for i in range(10):
        filepath = temp_dir / f"mp_{i:03d}.tif"
        _create_test_image(filepath, (20, 20), dtype=np.uint8)

    # Test with multiprocessing enabled
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        num_workers=2,
    )

    assert len(result) == 1
    assert result["mp"]["file_count"] == 10
    assert result["mp"]["shape"] == (10, 20, 20)

    # Verify zarr file was created correctly
    zarr_path = Path(result["mp"]["zarr_path"])
    assert zarr_path.exists()

    z = zarr.open(str(zarr_path), mode="r")
    assert z.shape == (10, 20, 20)
    assert z.dtype == np.uint8


def test_stack_files_to_zarr_multiprocessing_disabled(temp_dir):
    """Test that num_workers=1 disables multiprocessing."""
    # Create multiple images
    for i in range(5):
        filepath = temp_dir / f"seq_{i:03d}.tif"
        _create_test_image(filepath, (15, 15), dtype=np.uint8)

    # Test with multiprocessing disabled
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        num_workers=1,  # Disable multiprocessing
    )

    assert len(result) == 1
    assert result["seq"]["file_count"] == 5
    assert result["seq"]["shape"] == (5, 15, 15)


def test_stack_files_to_zarr_multiprocessing_multi_channel(temp_dir):
    """Test multiprocessing with multi-channel images."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create multi-channel images
    for i in range(8):
        filepath = temp_dir / f"mcmp_{i:02d}.tif"
        data = np.random.randint(0, 255, size=(3, 16, 16), dtype=np.uint8)
        tifffile.imwrite(str(filepath), data)

    # Test with multiprocessing
    result = stack_files_to_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        num_workers=3,
    )

    assert len(result) == 1
    assert result["mcmp"]["file_count"] == 8
    assert result["mcmp"]["shape"] == (8, 3, 16, 16)  # ZCYX order


# ============================================================================
# Tests for stack_files_to_ome_zarr()
# ============================================================================


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_single_channel(temp_dir):
    """Test OME-Zarr creation with single channel images."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(5):
        filepath = temp_dir / f"stack_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    # Convert to OME-Zarr
    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=3,
        downsample_mode="2d",  # Don't downsample Z
    )

    assert len(result) == 1
    assert "stack" in result

    metadata = result["stack"]
    assert metadata["file_count"] == 5
    assert metadata["shape"] == (5, 64, 64)  # (Z, Y, X)
    assert metadata["pyramid_levels"] == 3

    # Check OME-Zarr file exists and has correct structure
    zarr_path = Path(metadata["zarr_path"])
    assert zarr_path.exists()
    assert zarr_path.name.endswith(".ome.zarr")

    # Open and check structure
    root = zarr.open_group(str(zarr_path), mode="r")
    
    # Check that pyramid levels exist
    for level in range(3):
        assert str(level) in root, f"Pyramid level {level} not found"
    
    # Check base level shape
    assert root["0"].shape == (5, 64, 64)
    
    # Check multiscales metadata
    assert "multiscales" in root.attrs
    assert len(root.attrs["multiscales"]) > 0


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_downsample_mode_2d(temp_dir):
    """Test OME-Zarr with 2D downsampling mode (no Z downsampling)."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(10):
        filepath = temp_dir / f"test_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(128, 128), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=3,
        downsample_mode="2d",
    )

    metadata = result["test"]
    zarr_path = Path(metadata["zarr_path"])
    root = zarr.open_group(str(zarr_path), mode="r")

    # Check that Z dimension stays same across levels in 2D mode
    base_z = root["0"].shape[0]
    for level in range(1, 3):
        level_z = root[str(level)].shape[0]
        assert level_z == base_z, f"Z dimension changed in 2D mode: {base_z} -> {level_z}"


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_downsample_mode_3d(temp_dir):
    """Test OME-Zarr with 3D downsampling mode (downsample Z, Y, X)."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(8):
        filepath = temp_dir / f"vol_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=2,
        downsample_mode="3d",
    )

    metadata = result["vol"]
    zarr_path = Path(metadata["zarr_path"])
    root = zarr.open_group(str(zarr_path), mode="r")

    # In 3D mode, Z should be downsampled
    base_shape = root["0"].shape
    level1_shape = root["1"].shape
    
    # Z dimension should be downsampled (if original is divisible by 2)
    # Allow for rounding differences
    assert level1_shape[0] <= base_shape[0] // 2 + 1


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_custom_axes(temp_dir):
    """Test OME-Zarr with custom downsample_axes."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(5):
        filepath = temp_dir / f"custom_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(128, 128), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=2,
        downsample_axes=("y", "x"),  # Explicitly specify only Y, X
    )

    metadata = result["custom"]
    zarr_path = Path(metadata["zarr_path"])
    root = zarr.open_group(str(zarr_path), mode="r")

    # Z dimension should remain unchanged
    base_z = root["0"].shape[0]
    level1_z = root["1"].shape[0]
    assert level1_z == base_z


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_dask_method(temp_dir):
    """Test OME-Zarr with dask_coarsen method."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(3):
        filepath = temp_dir / f"dask_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=2,
        downsample_method="dask_coarsen",
    )

    assert len(result) == 1
    assert "dask" in result


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_multiscales_metadata(temp_dir):
    """Test that OME-Zarr has correct multiscales metadata structure."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(3):
        filepath = temp_dir / f"meta_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(32, 32), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=2,
    )

    metadata = result["meta"]
    zarr_path = Path(metadata["zarr_path"])
    root = zarr.open_group(str(zarr_path), mode="r")

    # Check multiscales metadata
    assert "multiscales" in root.attrs
    multiscales = root.attrs["multiscales"]
    assert len(multiscales) > 0
    assert "version" in multiscales[0]
    assert "axes" in multiscales[0]
    assert "datasets" in multiscales[0]
    
    # Check OME metadata
    assert "omero" in root.attrs


# ============================================================================
# Tests for stack_files_to_ome_zarr()
# ============================================================================


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_single_channel(temp_dir):
    """Test OME-Zarr creation with single channel images."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(5):
        filepath = temp_dir / f"stack_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    # Convert to OME-Zarr
    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=3,
        downsample_mode="2d",  # Don't downsample Z
    )

    assert len(result) == 1
    assert "stack" in result

    metadata = result["stack"]
    assert metadata["file_count"] == 5
    assert metadata["shape"] == (5, 64, 64)  # (Z, Y, X)
    assert metadata["pyramid_levels"] == 3

    # Check OME-Zarr file exists and has correct structure
    zarr_path = Path(metadata["zarr_path"])
    assert zarr_path.exists()
    assert zarr_path.name.endswith(".ome.zarr")

    # Open and check structure
    root = zarr.open_group(str(zarr_path), mode="r")
    
    # Check that pyramid levels exist
    for level in range(3):
        assert str(level) in root, f"Pyramid level {level} not found"
    
    # Check base level shape
    assert root["0"].shape == (5, 64, 64)
    
    # Check multiscales metadata
    assert "multiscales" in root.attrs
    assert len(root.attrs["multiscales"]) > 0


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_downsample_mode_2d(temp_dir):
    """Test OME-Zarr with 2D downsampling mode (no Z downsampling)."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(10):
        filepath = temp_dir / f"test_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(128, 128), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=3,
        downsample_mode="2d",
    )

    metadata = result["test"]
    zarr_path = Path(metadata["zarr_path"])
    root = zarr.open_group(str(zarr_path), mode="r")

    # Check that Z dimension stays same across levels in 2D mode
    base_z = root["0"].shape[0]
    for level in range(1, 3):
        level_z = root[str(level)].shape[0]
        assert level_z == base_z, f"Z dimension changed in 2D mode: {base_z} -> {level_z}"


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_downsample_mode_3d(temp_dir):
    """Test OME-Zarr with 3D downsampling mode (downsample Z, Y, X)."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(8):
        filepath = temp_dir / f"vol_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=2,
        downsample_mode="3d",
    )

    metadata = result["vol"]
    zarr_path = Path(metadata["zarr_path"])
    root = zarr.open_group(str(zarr_path), mode="r")

    # In 3D mode, Z should be downsampled
    base_shape = root["0"].shape
    level1_shape = root["1"].shape
    
    # Z dimension should be downsampled (if original is divisible by 2)
    # Allow for rounding differences
    assert level1_shape[0] <= base_shape[0] // 2 + 1


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_custom_axes(temp_dir):
    """Test OME-Zarr with custom downsample_axes."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(5):
        filepath = temp_dir / f"custom_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(128, 128), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=2,
        downsample_axes=("y", "x"),  # Explicitly specify only Y, X
    )

    metadata = result["custom"]
    zarr_path = Path(metadata["zarr_path"])
    root = zarr.open_group(str(zarr_path), mode="r")

    # Z dimension should remain unchanged
    base_z = root["0"].shape[0]
    level1_z = root["1"].shape[0]
    assert level1_z == base_z


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_dask_method(temp_dir):
    """Test OME-Zarr with dask_coarsen method."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(3):
        filepath = temp_dir / f"dask_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=2,
        downsample_method="dask_coarsen",
    )

    assert len(result) == 1
    assert "dask" in result


@pytest.mark.skipif(zarr is None, reason="zarr not available")
def test_stack_files_to_ome_zarr_multiscales_metadata(temp_dir):
    """Test that OME-Zarr has correct multiscales metadata structure."""
    if tifffile is None:
        pytest.skip("tifffile not available")

    # Create test images
    for i in range(3):
        filepath = temp_dir / f"meta_{i:03d}.tif"
        data = np.random.randint(0, 65535, size=(32, 32), dtype=np.uint16)
        tifffile.imwrite(str(filepath), data)

    result = stack_files_to_ome_zarr(
        directory=temp_dir,
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        pyramid_levels=2,
    )

    metadata = result["meta"]
    zarr_path = Path(metadata["zarr_path"])
    root = zarr.open_group(str(zarr_path), mode="r")

    # Check multiscales metadata
    assert "multiscales" in root.attrs
    multiscales = root.attrs["multiscales"]
    assert len(multiscales) > 0
    assert "version" in multiscales[0]
    assert "axes" in multiscales[0]
    assert "datasets" in multiscales[0]
    
    # Check OME metadata
    assert "omero" in root.attrs
