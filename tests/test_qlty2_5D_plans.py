"""
Tests for 2.5D Quilt plan system.
"""

import torch

from qlty.qlty2_5D import (
    ExtractionPlan,
    NCZYX25DQuilt,
    PatchExtraction,
    StitchingPlan,
)


def test_create_extraction_plan_simple():
    """Test creating extraction plan without 2D quilting."""
    data = torch.zeros(2, 1, 5, 10, 10)  # 2 images, 1 channel, 5 z-slices

    spec = {"identity": [0]}

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="3d",
        z_slices=[0, 1, 2],
    )

    plan = quilt.create_extraction_plan()

    # Should have 2 images * 3 z-slices = 6 patches
    assert plan.total_patches == 6
    assert len(plan.patches) == 6

    # Check first patch
    patch0 = plan.patches[0]
    assert patch0.n == 0
    assert patch0.z0 == 0
    assert patch0.y_start is None  # No 2D quilting
    assert patch0.color_y_idx == 0
    assert patch0.color_x_idx == 0


def test_create_extraction_plan_with_2d_quilting():
    """Test creating extraction plan with 2D quilting."""
    data = torch.zeros(1, 1, 5, 20, 20)

    spec = {"identity": [0]}

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[0],
    )

    plan = quilt.create_extraction_plan(window=(10, 10), step=(5, 5))

    # Should have patches for 2D quilting
    assert plan.total_patches > 1

    # Check that patches have spatial coordinates
    patch0 = plan.patches[0]
    assert patch0.y_start is not None
    assert patch0.y_stop is not None
    assert patch0.x_start is not None
    assert patch0.x_stop is not None


def test_create_extraction_plan_color_groups():
    """Test color group assignments in extraction plan."""
    data = torch.zeros(1, 1, 5, 20, 20)

    spec = {"identity": [0]}

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[0],
    )

    plan = quilt.create_extraction_plan(
        window=(10, 10),
        step=(5, 5),
        color_y_mod=2,
        color_x_mod=2,
    )

    # Check that color groups are created
    assert len(plan.color_groups) > 0

    # Check that patches are assigned to color groups
    for patch in plan.patches:
        color_key = (patch.color_y_idx, patch.color_x_idx)
        assert color_key in plan.color_groups
        assert patch.patch_idx in plan.color_groups[color_key]


def test_get_patches_for_color():
    """Test getting patches for a specific color group."""
    data = torch.zeros(1, 1, 5, 20, 20)

    spec = {"identity": [0]}

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[0],
    )

    plan = quilt.create_extraction_plan(
        window=(10, 10),
        step=(5, 5),
        color_y_mod=2,
        color_x_mod=2,
    )

    # Get patches for color group (0, 0)
    patches = plan.get_patches_for_color(0, 0)

    assert len(patches) > 0
    for patch in patches:
        assert patch.color_y_idx == 0
        assert patch.color_x_idx == 0


def test_create_stitching_plan():
    """Test creating stitching plan."""
    data = torch.zeros(1, 1, 5, 20, 20)

    spec = {"identity": [0]}

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="3d",
        z_slices=[0, 1, 2],
    )

    extraction_plan = quilt.create_extraction_plan()
    stitching_plan = quilt.create_stitching_plan(extraction_plan)

    # Check output shape
    assert stitching_plan.output_shape == (1, 1, 3, 20, 20)

    # Check patch mappings
    assert len(stitching_plan.patch_mappings) == extraction_plan.total_patches

    # Check color groups match
    assert stitching_plan.color_groups == extraction_plan.color_groups


def test_plan_serialization():
    """Test plan serialization and deserialization."""
    data = torch.zeros(1, 1, 5, 10, 10)

    spec = {"identity": [0]}

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="3d",
        z_slices=[0, 1],
    )

    extraction_plan = quilt.create_extraction_plan()

    # Serialize
    serialized = extraction_plan.serialize()

    # Check serialized format
    assert "patches" in serialized
    assert "color_groups" in serialized
    assert "total_patches" in serialized

    # Deserialize
    deserialized = ExtractionPlan.deserialize(serialized)

    # Check that deserialized plan matches original
    assert deserialized.total_patches == extraction_plan.total_patches
    assert len(deserialized.patches) == len(extraction_plan.patches)
    assert len(deserialized.color_groups) == len(extraction_plan.color_groups)


def test_stitching_plan_serialization():
    """Test stitching plan serialization and deserialization."""
    data = torch.zeros(1, 1, 5, 10, 10)

    spec = {"identity": [0]}

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[0],
    )

    extraction_plan = quilt.create_extraction_plan()
    stitching_plan = quilt.create_stitching_plan(extraction_plan)

    # Serialize
    serialized = stitching_plan.serialize()

    # Check serialized format
    assert "output_shape" in serialized
    assert "patch_mappings" in serialized
    assert "color_groups" in serialized

    # Deserialize
    deserialized = StitchingPlan.deserialize(serialized)

    # Check that deserialized plan matches original
    assert deserialized.output_shape == stitching_plan.output_shape
    assert len(deserialized.patch_mappings) == len(stitching_plan.patch_mappings)
    assert len(deserialized.color_groups) == len(stitching_plan.color_groups)


def test_required_z_indices():
    """Test that required_z_indices are computed correctly."""
    data = torch.zeros(1, 1, 10, 20, 20)

    spec = {"identity": [-1, 0, 1], "mean": [[-2, -3], [2, 3]]}

    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="3d",
        z_slices=[5],  # Center at z=5
    )

    plan = quilt.create_extraction_plan()

    # Check that required_z_indices include all needed slices
    patch = plan.patches[0]
    required_z = set(patch.required_z_indices)

    # Should include z=5 (center) and neighbors
    assert 5 in required_z
    # Should include z=4, z=6 (for direct -1, +1)
    assert 4 in required_z or 6 in required_z
    # Should include slices for mean operations too


if __name__ == "__main__":
    test_create_extraction_plan_simple()

    test_create_extraction_plan_with_2d_quilting()

    test_create_extraction_plan_color_groups()

    test_get_patches_for_color()

    test_create_stitching_plan()

    test_plan_serialization()

    test_stitching_plan_serialization()

    test_required_z_indices()
