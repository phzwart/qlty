"""
Additional tests for qlty2_5D to improve coverage.
Focuses on error handling, edge cases, and uncovered code paths.
"""

import pytest
import torch

from qlty.qlty2_5D import (
    NCZYX25DQuilt,
    ZOperation,
    apply_boundary_mode,
    compute_channel_count,
    parse_channel_spec,
)


def test_parse_channel_spec_unknown_string_key():
    """Test error with unknown string operation key."""
    with pytest.raises(ValueError, match="Unknown operation"):
        parse_channel_spec({"unknown_op": [1, 2, 3]})


def test_parse_channel_spec_invalid_key_type():
    """Test error with invalid key type."""
    with pytest.raises(ValueError, match="Invalid key type"):
        parse_channel_spec({123: [1, 2, 3]})


def test_parse_channel_spec_identity_invalid_value_type():
    """Test error when IDENTITY operation gets wrong value type."""
    with pytest.raises(ValueError, match="IDENTITY operation requires"):
        parse_channel_spec({"identity": "invalid"})


def test_parse_channel_spec_identity_empty_offsets():
    """Test error when IDENTITY operation has empty offsets."""
    with pytest.raises(ValueError, match="must have at least one offset"):
        parse_channel_spec({"identity": []})


def test_parse_channel_spec_mean_invalid_value_type():
    """Test error when MEAN operation gets wrong value type."""
    with pytest.raises(ValueError, match="MEAN operation requires"):
        parse_channel_spec({"mean": "invalid"})


def test_parse_channel_spec_mean_empty_groups():
    """Test error when MEAN operation has no offset groups."""
    with pytest.raises(ValueError, match="must have at least one offset group"):
        parse_channel_spec({"mean": []})


def test_parse_channel_spec_mean_empty_group():
    """Test error when MEAN operation has empty offset group."""
    with pytest.raises(ValueError, match="offset groups cannot be empty"):
        parse_channel_spec({"mean": [[]]})


def test_parse_channel_spec_mean_invalid_group_type():
    """Test error when MEAN operation has invalid group type."""
    with pytest.raises(ValueError, match="offset groups must be lists"):
        parse_channel_spec({"mean": [123]})


def test_parse_channel_spec_std_invalid_value_type():
    """Test error when STD operation gets wrong value type."""
    with pytest.raises(ValueError, match="STD operation requires"):
        parse_channel_spec({"std": "invalid"})


def test_parse_channel_spec_std_empty_groups():
    """Test error when STD operation has no offset groups."""
    with pytest.raises(ValueError, match="must have at least one offset group"):
        parse_channel_spec({"std": []})


def test_parse_channel_spec_std_empty_group():
    """Test error when STD operation has empty offset group."""
    with pytest.raises(ValueError, match="offset groups cannot be empty"):
        parse_channel_spec({"std": [[]]})


def test_parse_channel_spec_std_invalid_group_type():
    """Test error when STD operation has invalid group type."""
    with pytest.raises(ValueError, match="offset groups must be lists"):
        parse_channel_spec({"std": [123]})


def test_parse_channel_spec_unknown_operation_type():
    """Test error with unknown operation type."""
    # This would require creating a custom ZOperation, which is difficult
    # So we test via invalid string key instead
    with pytest.raises(ValueError, match="Unknown operation"):
        parse_channel_spec({"nonexistent": [1, 2]})


def test_apply_boundary_mode_zero():
    """Test boundary mode zero (returns -1)."""
    assert apply_boundary_mode(-1, 0, 10, "zero") == -1
    assert apply_boundary_mode(10, 0, 10, "zero") == -1
    assert apply_boundary_mode(5, 0, 10, "zero") == 5


def test_apply_boundary_mode_skip():
    """Test boundary mode skip (returns -1)."""
    assert apply_boundary_mode(-1, 0, 10, "skip") == -1
    assert apply_boundary_mode(10, 0, 10, "skip") == -1
    assert apply_boundary_mode(5, 0, 10, "skip") == 5


def test_apply_boundary_mode_reflect_edge_cases():
    """Test boundary mode reflect with various edge cases."""
    # Test at boundaries
    assert apply_boundary_mode(0, 0, 10, "reflect") == 0
    assert apply_boundary_mode(9, 0, 10, "reflect") == 9
    # Test just outside
    assert apply_boundary_mode(-1, 0, 10, "reflect") == 0
    assert apply_boundary_mode(10, 0, 10, "reflect") == 9
    # Test far outside
    assert apply_boundary_mode(-5, 0, 10, "reflect") == 4
    assert apply_boundary_mode(15, 0, 10, "reflect") == 4


def test_apply_boundary_mode_unknown():
    """Test error with unknown boundary mode."""
    # Need to use an index that's out of bounds to trigger the boundary mode check
    with pytest.raises(ValueError, match="Unknown boundary_mode"):
        apply_boundary_mode(-1, 0, 10, "invalid_mode")


def test_nczyx25dquilt_invalid_data_source_type():
    """Test error with invalid data source type."""
    with pytest.raises(TypeError, match="must be torch.Tensor or TensorLike3D"):
        NCZYX25DQuilt(data_source="invalid", channel_spec={"identity": [0]})


def test_nczyx25dquilt_invalid_shape():
    """Test error with non-5D data source."""
    data = torch.randn(1, 2, 3, 4)  # 4D instead of 5D
    with pytest.raises(ValueError, match="must be 5D"):
        NCZYX25DQuilt(data_source=data, channel_spec={"identity": [0]})


def test_nczyx25dquilt_invalid_accumulation_mode():
    """Test error with invalid accumulation mode."""
    data = torch.randn(1, 1, 5, 10, 10)
    with pytest.raises(ValueError, match="accumulation_mode must be"):
        NCZYX25DQuilt(
            data_source=data,
            channel_spec={"identity": [0]},
            accumulation_mode="invalid",
        )


def test_nczyx25dquilt_invalid_boundary_mode():
    """Test error with invalid boundary mode."""
    data = torch.randn(1, 1, 5, 10, 10)
    with pytest.raises(ValueError, match="boundary_mode must be"):
        NCZYX25DQuilt(
            data_source=data, channel_spec={"identity": [0]}, boundary_mode="invalid"
        )


def test_nczyx25dquilt_invalid_z_slices_type():
    """Test error with invalid z_slices type."""
    data = torch.randn(1, 1, 5, 10, 10)
    with pytest.raises(ValueError, match="z_slices must be slice"):
        NCZYX25DQuilt(
            data_source=data, channel_spec={"identity": [0]}, z_slices="invalid"
        )


def test_nczyx25dquilt_invalid_z_index():
    """Test error with z_slices containing invalid index."""
    data = torch.randn(1, 1, 5, 10, 10)
    with pytest.raises(ValueError, match="z_slices contains invalid index"):
        NCZYX25DQuilt(data_source=data, channel_spec={"identity": [0]}, z_slices=[10])


def test_nczyx25dquilt_empty_z_slices():
    """Test error with z_slices resulting in empty list."""
    data = torch.randn(1, 1, 5, 10, 10)
    with pytest.raises(ValueError, match="results in empty list"):
        NCZYX25DQuilt(
            data_source=data, channel_spec={"identity": [0]}, z_slices=slice(10, 5)
        )


def test_std_operation():
    """Test STD operation in channel spec."""
    data = torch.randn(1, 1, 10, 20, 20)

    spec = {"std": [[-1, -2], [1, 2]]}
    quilt = NCZYX25DQuilt(data_source=data, channel_spec=spec, accumulation_mode="2d")

    result = quilt.convert()
    # 2 std operations * 1 input channel = 2 output channels
    # 10 z-slices in 2d mode = 10 separate 2D images
    assert result.shape == (10, 2, 20, 20)


def test_boundary_mode_zero():
    """Test boundary mode zero with conversion."""
    data = torch.zeros(1, 1, 5, 10, 10)
    # Set known values
    data[0, 0, 2] = 1.0  # z=2

    spec = {"identity": [-2, -1, 0]}  # Will go out of bounds for z_slices=[2]
    quilt = NCZYX25DQuilt(
        data_source=data, channel_spec=spec, boundary_mode="zero", z_slices=[2]
    )

    result = quilt.convert()
    assert result.shape == (1, 3, 10, 10)
    # Out-of-bounds channels should be zero (z=-2 and z=-1 when z0=2 means z=0,1)
    # Channel 0: z=-2 (when z0=2, means z=0) -> should be zero
    assert torch.all(result[0, 0] == 0)  # First channel should be zero (out of bounds)
    # Channel 1: z=-1 (when z0=2, means z=1) -> should be zero
    assert torch.all(result[0, 1] == 0)  # Second channel should be zero (out of bounds)
    # Channel 2: z=0 (when z0=2, means z=2) -> should be 1.0
    assert torch.allclose(result[0, 2], data[0, 0, 2])


def test_boundary_mode_skip():
    """Test boundary mode skip with conversion."""
    data = torch.randn(1, 1, 5, 10, 10)

    spec = {"identity": [-2, -1, 0]}  # Some will go out of bounds
    quilt = NCZYX25DQuilt(
        data_source=data, channel_spec=spec, boundary_mode="skip", z_slices=[2]
    )

    result = quilt.convert()
    # With skip mode, invalid channels are skipped
    assert result.shape[0] == 1
    # Should have fewer channels than requested due to skipping


def test_boundary_mode_reflect_operation():
    """Test boundary mode reflect in operations."""
    data = torch.arange(1 * 1 * 5 * 10 * 10).reshape(1, 1, 5, 10, 10).float()

    spec = {"identity": [-1, 0, 1]}
    quilt = NCZYX25DQuilt(
        data_source=data, channel_spec=spec, boundary_mode="reflect", z_slices=[0]
    )

    result = quilt.convert()
    assert result.shape == (1, 3, 10, 10)


def test_mean_operation_single_slice_group():
    """Test MEAN operation with single slice groups."""
    data = torch.randn(1, 1, 10, 20, 20)

    spec = {"mean": [[0], [1], [2]]}  # Each group has one slice (mean of one)
    quilt = NCZYX25DQuilt(data_source=data, channel_spec=spec, accumulation_mode="2d")

    result = quilt.convert()
    assert result.shape == (10, 3, 20, 20)


def test_operation_all_skipped_error():
    """Test error when all operations are skipped."""
    data = torch.randn(1, 1, 5, 10, 10)

    spec = {"identity": [-10]}  # Way out of bounds
    quilt = NCZYX25DQuilt(
        data_source=data, channel_spec=spec, boundary_mode="skip", z_slices=[2]
    )

    # This should raise an error when all channels are skipped
    with pytest.raises(ValueError, match="produced no valid channels"):
        quilt.convert()


def test_z_slices_slice_object():
    """Test z_slices with slice object."""
    data = torch.randn(1, 1, 10, 20, 20)

    spec = {"identity": [0]}
    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        z_slices=slice(2, 7),  # z-slices 2-6
    )

    result = quilt.convert()
    # In 2d mode, 5 z-slices become 5 separate 2D images
    assert result.shape == (5, 1, 20, 20)


def test_z_slices_tuple():
    """Test z_slices with tuple (should work like list)."""
    data = torch.randn(1, 1, 10, 20, 20)

    spec = {"identity": [0]}
    quilt = NCZYX25DQuilt(data_source=data, channel_spec=spec, z_slices=(0, 2, 4))

    result = quilt.convert()
    assert result.shape == (3, 1, 20, 20)


def test_group_by_operation():
    """Test group_by_operation parameter."""
    data = torch.randn(1, 1, 10, 20, 20)

    spec = {"identity": [0], "mean": [[-1, 0, 1]]}
    quilt = NCZYX25DQuilt(
        data_source=data, channel_spec=spec, group_by_operation=True, z_slices=[5]
    )

    result = quilt.convert()
    # Should still produce same number of channels, just grouped differently
    assert result.shape == (1, 2, 20, 20)  # 1 identity + 1 mean = 2 channels


def test_convert_3d_mode():
    """Test conversion in 3d accumulation mode."""
    data = torch.randn(2, 3, 5, 10, 10)

    spec = {"identity": [0, 1]}
    quilt = NCZYX25DQuilt(
        data_source=data, channel_spec=spec, accumulation_mode="3d", z_slices=[1, 2, 3]
    )

    result = quilt.convert()
    # In 3d mode: (N, C', Z_selected, Y, X)
    # N=2, C'=2*3=6 (2 identity channels * 3 input channels), Z=3, Y=10, X=10
    assert result.shape == (2, 6, 3, 10, 10)


def test_convert_with_std_operation():
    """Test conversion with STD operation."""
    data = torch.randn(1, 1, 10, 20, 20)

    spec = {"std": [[-1, 0, 1], [-2, -1, 0, 1, 2]]}
    quilt = NCZYX25DQuilt(data_source=data, channel_spec=spec, z_slices=[5])

    result = quilt.convert()
    assert result.shape == (1, 2, 20, 20)


def test_multiple_input_channels():
    """Test conversion with multiple input channels."""
    data = torch.randn(1, 3, 10, 20, 20)  # 3 input channels

    spec = {"identity": [0, 1]}
    quilt = NCZYX25DQuilt(data_source=data, channel_spec=spec, z_slices=[5])

    result = quilt.convert()
    # 2 identity channels * 3 input channels = 6 output channels
    assert result.shape == (1, 6, 20, 20)


def test_get_required_z_range():
    """Test get_required_z_range method of ChannelOperation."""
    from qlty.qlty2_5D import ChannelOperation

    # IDENTITY operation
    op = ChannelOperation(
        op_type=ZOperation.IDENTITY, offsets=(-1, 0, 1), output_channels=3
    )
    z_min, z_max = op.get_required_z_range(z0=5)
    assert z_min == 4  # 5 + (-1)
    assert z_max == 7  # 5 + 1 + 1 (exclusive: max(5-1, 5+0, 5+1) + 1 = 6 + 1 = 7)

    # MEAN operation
    op_mean = ChannelOperation(
        op_type=ZOperation.MEAN,
        offsets=((-2, -1, 0), (0, 1, 2)),
        output_channels=2,
    )
    z_min, z_max = op_mean.get_required_z_range(z0=5)
    assert z_min == 3  # 5 + (-2)
    assert z_max == 8  # 5 + 2 + 1 (exclusive: max of all offsets + 1)
