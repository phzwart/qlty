"""
Tests for 2.5D Quilt implementation.
"""

import torch

from qlty.qlty2_5D import (
    NCZYX25DQuilt,
    ZOperation,
    apply_boundary_mode,
    compute_channel_count,
    parse_channel_spec,
)


def test_parse_channel_spec_string():
    """Test parsing channel spec with string keys."""
    spec = {"identity": [-1, 0, 1], "mean": [[-1, -2, -3], [1, 2, 3]]}
    operations = parse_channel_spec(spec)

    assert len(operations) == 2
    assert operations[0].op_type == ZOperation.IDENTITY
    assert operations[0].output_channels == 3
    assert operations[1].op_type == ZOperation.MEAN
    assert operations[1].output_channels == 2


def test_parse_channel_spec_enum():
    """Test parsing channel spec with enum keys."""
    spec = {ZOperation.IDENTITY: (-1, 0, 1), ZOperation.MEAN: ((-1, -2, -3), (1, 2, 3))}
    operations = parse_channel_spec(spec)

    assert len(operations) == 2
    assert operations[0].op_type == ZOperation.IDENTITY
    assert operations[1].op_type == ZOperation.MEAN


def test_parse_channel_spec_empty():
    """Test that empty spec raises error."""
    import pytest

    with pytest.raises(ValueError, match="cannot be empty"):
        parse_channel_spec({})


def test_apply_boundary_mode_clamp():
    """Test boundary mode clamping."""
    # Within bounds
    assert apply_boundary_mode(5, 0, 10, "clamp") == 5
    # Below bounds
    assert apply_boundary_mode(-1, 0, 10, "clamp") == 0
    # Above bounds
    assert apply_boundary_mode(10, 0, 10, "clamp") == 9


def test_apply_boundary_mode_reflect():
    """Test boundary mode reflection."""
    # Within bounds
    assert apply_boundary_mode(5, 0, 10, "reflect") == 5
    # Below bounds - should reflect
    assert apply_boundary_mode(-1, 0, 10, "reflect") == 0
    assert apply_boundary_mode(-2, 0, 10, "reflect") == 1
    # Above bounds - should reflect
    assert apply_boundary_mode(10, 0, 10, "reflect") == 9
    assert apply_boundary_mode(11, 0, 10, "reflect") == 8


def test_compute_channel_count():
    """Test channel count computation."""
    spec = {"identity": [-1, 0, 1], "mean": [[-1, -2], [1, 2]]}
    operations = parse_channel_spec(spec)

    # 3 direct + 2 mean = 5 channels per input channel
    assert compute_channel_count(operations, 1) == 5
    assert compute_channel_count(operations, 3) == 15


def test_basic_conversion_2d():
    """Test basic 2.5D conversion in 2D mode."""
    # Create simple test data: (1, 1, 5, 10, 10) - 1 image, 1 channel, 5 z-slices
    data = torch.arange(1 * 1 * 5 * 10 * 10).reshape(1, 1, 5, 10, 10).float()

    spec = {
        "identity": [0],  # Just current slice
    }

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[0],  # Only process z=0
    )

    result = quilt.convert()

    # Should be (1, 1, 10, 10) - 1 image, 1 output channel, Y, X
    assert result.shape == (1, 1, 10, 10)

    # Result should match input at z=0
    assert torch.allclose(result[0, 0], data[0, 0, 0])


def test_basic_conversion_3d():
    """Test basic 2.5D conversion in 3D mode."""
    # Create simple test data: (1, 1, 5, 10, 10)
    data = torch.arange(1 * 1 * 5 * 10 * 10).reshape(1, 1, 5, 10, 10).float()

    spec = {
        "identity": [0],  # Just current slice
    }

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="3d",
        z_slices=[0, 1, 2],
    )

    result = quilt.convert()

    # Should be (1, 1, 3, 10, 10) - 1 image, 1 output channel, 3 z-slices, Y, X
    assert result.shape == (1, 1, 3, 10, 10)

    # Result should match input
    assert torch.allclose(result[0, 0, 0], data[0, 0, 0])
    assert torch.allclose(result[0, 0, 1], data[0, 0, 1])
    assert torch.allclose(result[0, 0, 2], data[0, 0, 2])


def test_multiple_operations():
    """Test conversion with multiple operations."""
    data = torch.arange(1 * 1 * 10 * 20 * 20).reshape(1, 1, 10, 20, 20).float()

    spec = {"identity": [-1, 0, 1], "mean": [[-1, -2], [1, 2]]}

    quilt = NCZYX25DQuilt(data_source=data, channel_spec=spec, accumulation_mode="2d")

    result = quilt.convert()

    # 3 direct + 2 mean = 5 channels per input channel
    # 1 input channel * 5 = 5 output channels
    # In 2d mode, each z-slice becomes a separate 2D image: (10, 5, 20, 20)
    assert result.shape == (10, 5, 20, 20)


def test_selective_z_slices():
    """Test selective z-slice processing."""
    data = torch.arange(1 * 1 * 10 * 10 * 10).reshape(1, 1, 10, 10, 10).float()

    spec = {"identity": [0]}

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="3d",
        z_slices=[0, 2, 4, 6, 8],
    )

    result = quilt.convert()

    # Should have 5 z-slices in output
    assert result.shape == (1, 1, 5, 10, 10)


def test_get_channel_metadata():
    """Test channel metadata generation."""
    data = torch.zeros(1, 2, 10, 20, 20)  # 2 input channels

    spec = {"identity": [-1, 0, 1], "mean": [[-1, -2]]}

    quilt = NCZYX25DQuilt(data_source=data, channel_spec=spec)
    metadata = quilt.get_channel_metadata()

    # 3 direct + 1 mean = 4 channels per input channel
    # 2 input channels * 4 = 8 total channels
    assert len(metadata) == 8

    # Check first channel metadata
    assert metadata[0]["input_channel"] == 0
    assert metadata[0]["operation_type"] == "IDENTITY"
    assert metadata[0]["offsets"] == (-1,)


def test_validate_spec():
    """Test specification validation."""
    data = torch.zeros(1, 1, 10, 20, 20)

    spec = {"identity": [0, 1, 2]}

    quilt = NCZYX25DQuilt(data_source=data, channel_spec=spec)
    is_valid, messages = quilt.validate_spec()

    assert is_valid
    assert len(messages) == 0  # No errors or warnings for valid spec


if __name__ == "__main__":
    # Run basic tests
    test_parse_channel_spec_string()

    test_parse_channel_spec_enum()

    test_apply_boundary_mode_clamp()

    test_apply_boundary_mode_reflect()

    test_compute_channel_count()

    test_basic_conversion_2d()

    test_basic_conversion_3d()

    test_multiple_operations()

    test_selective_z_slices()

    test_get_channel_metadata()

    test_validate_spec()
