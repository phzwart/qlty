#!/usr/bin/env python

"""Tests for positional embeddings functionality in patch extraction."""

import tempfile
from pathlib import Path

import pytest
import torch

try:
    import numpy as np
except ImportError:
    np = None

try:
    import zarr
except ImportError:
    zarr = None

from qlty.patch_pairs_2d import extract_patch_pairs

# Optional imports - may not exist in all versions
try:
    from qlty.patch_pairs_2d import (
        ZarrPatchPairDataset,
        extract_patches_from_metadata,
        extract_patches_to_zarr,
    )

    HAS_ADVANCED_FEATURES = True
except ImportError:
    HAS_ADVANCED_FEATURES = False
    ZarrPatchPairDataset = None
    extract_patches_from_metadata = None
    extract_patches_to_zarr = None

try:
    from qlty import qlty2D, qlty2DLarge

    HAS_QUILT_FEATURES = True
except ImportError:
    HAS_QUILT_FEATURES = False
    qlty2D = None
    qlty2DLarge = None


# ============================================================================
# Tests for extract_patch_pairs() with positional embeddings
# ============================================================================


def test_extract_patch_pairs_positions_basic():
    """Test basic positional embeddings extraction (Y, X only)."""
    tensor = torch.randn(2, 3, 64, 64)
    window = (16, 16)
    num_patches = 5
    delta_range = (6.0, 10.0)

    patches1, patches2, deltas, rotations, positions1, positions2 = extract_patch_pairs(
        tensor,
        window,
        num_patches,
        delta_range,
        return_positions=True,
    )

    # Check output shapes
    assert patches1.shape == (2 * num_patches, 3, 16, 16)
    assert patches2.shape == (2 * num_patches, 3, 16, 16)
    assert deltas.shape == (2 * num_patches, 2)
    assert rotations.shape == (2 * num_patches,)
    assert positions1.shape == (2 * num_patches, 2)  # [Y, X]
    assert positions2.shape == (2 * num_patches, 2)  # [Y, X]

    # Check dtype
    assert positions1.dtype == torch.int64
    assert positions2.dtype == torch.int64

    # Check that positions are within image bounds
    Y, X = tensor.shape[2], tensor.shape[3]
    assert torch.all(positions1[:, 0] >= 0)  # Y >= 0
    assert torch.all(positions1[:, 1] >= 0)  # X >= 0
    assert torch.all(positions1[:, 0] + window[0] <= Y)  # Y + window_height <= Y
    assert torch.all(positions1[:, 1] + window[1] <= X)  # X + window_width <= X

    assert torch.all(positions2[:, 0] >= 0)
    assert torch.all(positions2[:, 1] >= 0)
    assert torch.all(positions2[:, 0] + window[0] <= Y)
    assert torch.all(positions2[:, 1] + window[1] <= X)


def test_extract_patch_pairs_positions_with_n():
    """Test positional embeddings with N (batch) index included."""
    tensor = torch.randn(3, 2, 64, 64)
    window = (16, 16)
    num_patches = 4
    delta_range = (6.0, 10.0)

    patches1, patches2, deltas, rotations, positions1, positions2 = extract_patch_pairs(
        tensor,
        window,
        num_patches,
        delta_range,
        return_positions=True,
        include_n_position=True,
    )

    # Check output shapes
    assert positions1.shape == (3 * num_patches, 3)  # [N, Y, X]
    assert positions2.shape == (3 * num_patches, 3)  # [N, Y, X]

    # Check that N indices are valid
    assert torch.all(positions1[:, 0] >= 0)
    assert torch.all(positions1[:, 0] < tensor.shape[0])
    assert torch.all(positions2[:, 0] >= 0)
    assert torch.all(positions2[:, 0] < tensor.shape[0])

    # Check that positions match deltas
    for i in range(len(positions1)):
        # patch2 position should be patch1 position + delta
        y1, x1 = positions1[i, 1].item(), positions1[i, 2].item()
        y2, x2 = positions2[i, 1].item(), positions2[i, 2].item()
        dy = deltas[i, 1].item()
        dx = deltas[i, 0].item()

        assert abs(y2 - y1 - dy) < 1e-6
        assert abs(x2 - x1 - dx) < 1e-6


def test_extract_patch_pairs_positions_reproducibility():
    """Test that positional embeddings are reproducible with same seed."""
    tensor = torch.randn(2, 2, 64, 64)
    window = (16, 16)
    num_patches = 5
    delta_range = (6.0, 10.0)

    result_a = extract_patch_pairs(
        tensor,
        window,
        num_patches,
        delta_range,
        random_seed=42,
        return_positions=True,
    )
    result_b = extract_patch_pairs(
        tensor,
        window,
        num_patches,
        delta_range,
        random_seed=42,
        return_positions=True,
    )

    patches1_a, patches2_a, deltas_a, rotations_a, positions1_a, positions2_a = result_a
    patches1_b, patches2_b, deltas_b, rotations_b, positions1_b, positions2_b = result_b

    # All positions should match
    assert torch.allclose(positions1_a.float(), positions1_b.float())
    assert torch.allclose(positions2_a.float(), positions2_b.float())


def test_extract_patch_pairs_positions_without_flag():
    """Test that positions are not returned when flag is False."""
    tensor = torch.randn(2, 3, 64, 64)
    window = (16, 16)
    num_patches = 5
    delta_range = (6.0, 10.0)

    result = extract_patch_pairs(
        tensor,
        window,
        num_patches,
        delta_range,
        return_positions=False,
    )

    # Should return 4 items (no positions)
    assert len(result) == 4
    patches1, patches2, deltas, rotations = result
    assert patches1.shape == (2 * num_patches, 3, 16, 16)


# ============================================================================
# Tests for extract_patches_from_metadata() with positional embeddings
# ============================================================================


@pytest.mark.skipif(not HAS_ADVANCED_FEATURES, reason="Advanced features not available")
def test_extract_patches_from_metadata_positions_basic():
    """Test positional embeddings extraction from metadata (Y, X only)."""
    from qlty.patch_pairs_2d import extract_patch_pairs_metadata

    tensor = torch.randn(2, 3, 64, 64)
    window = (16, 16)
    num_patches = 5
    delta_range = (6.0, 10.0)

    metadata = extract_patch_pairs_metadata(
        tensor,
        window,
        num_patches,
        delta_range,
        random_seed=42,
    )

    selected_indices = [0, 1, 2]

    (
        patches1,
        patches2,
        deltas,
        rotations,
        positions1,
        positions2,
    ) = extract_patches_from_metadata(
        tensor,
        metadata,
        selected_indices,
        return_positions=True,
    )

    # Check shapes
    assert positions1.shape == (3, 2)  # [Y, X]
    assert positions2.shape == (3, 2)  # [Y, X]

    # Check that positions match metadata
    for i, idx in enumerate(selected_indices):
        assert positions1[i, 0].item() == metadata["patch1_y"][idx].item()
        assert positions1[i, 1].item() == metadata["patch1_x"][idx].item()
        assert positions2[i, 0].item() == metadata["patch2_y"][idx].item()
        assert positions2[i, 1].item() == metadata["patch2_x"][idx].item()


@pytest.mark.skipif(not HAS_ADVANCED_FEATURES, reason="Advanced features not available")
def test_extract_patches_from_metadata_positions_with_n():
    """Test positional embeddings with N index from metadata."""
    from qlty.patch_pairs_2d import extract_patch_pairs_metadata

    tensor = torch.randn(3, 2, 64, 64)
    window = (16, 16)
    num_patches = 4
    delta_range = (6.0, 10.0)

    metadata = extract_patch_pairs_metadata(
        tensor,
        window,
        num_patches,
        delta_range,
        random_seed=42,
    )

    selected_indices = [0, 1, 2, 5]

    (
        patches1,
        patches2,
        deltas,
        rotations,
        positions1,
        positions2,
    ) = extract_patches_from_metadata(
        tensor,
        metadata,
        selected_indices,
        return_positions=True,
        include_n_position=True,
    )

    # Check shapes
    assert positions1.shape == (4, 3)  # [N, Y, X]
    assert positions2.shape == (4, 3)  # [N, Y, X]

    # Check that positions match metadata
    for i, idx in enumerate(selected_indices):
        assert positions1[i, 0].item() == metadata["image_idx"][idx].item()
        assert positions1[i, 1].item() == metadata["patch1_y"][idx].item()
        assert positions1[i, 2].item() == metadata["patch1_x"][idx].item()
        assert positions2[i, 0].item() == metadata["image_idx"][idx].item()
        assert positions2[i, 1].item() == metadata["patch2_y"][idx].item()
        assert positions2[i, 2].item() == metadata["patch2_x"][idx].item()


# ============================================================================
# Tests for extract_patches_to_zarr() with positional embeddings
# ============================================================================


@pytest.mark.skipif(zarr is None, reason="zarr not available")
@pytest.mark.skipif(not HAS_ADVANCED_FEATURES, reason="Advanced features not available")
def test_extract_patches_to_zarr_positions_basic():
    """Test storing positional embeddings in zarr (Y, X only)."""
    from qlty.patch_pairs_2d import extract_patch_pairs_metadata

    tensor = torch.randn(2, 3, 64, 64)
    window = (16, 16)
    num_patches = 5
    delta_range = (6.0, 10.0)

    metadata = extract_patch_pairs_metadata(
        tensor,
        window,
        num_patches,
        delta_range,
        random_seed=42,
    )

    selected_indices = [0, 1, 2]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = str(Path(tmpdir) / "patches.zarr")

        zarr_group = extract_patches_to_zarr(
            tensor,
            metadata,
            selected_indices,
            zarr_path,
            store_positions=True,
        )

        # Check that position arrays exist
        assert "positions1" in zarr_group
        assert "positions2" in zarr_group

        # Check shapes
        assert zarr_group["positions1"].shape == (3, 2)  # [Y, X]
        assert zarr_group["positions2"].shape == (3, 2)  # [Y, X]

        # Check dtype
        assert zarr_group["positions1"].dtype == "int64"
        assert zarr_group["positions2"].dtype == "int64"


@pytest.mark.skipif(zarr is None, reason="zarr not available")
@pytest.mark.skipif(not HAS_ADVANCED_FEATURES, reason="Advanced features not available")
def test_extract_patches_to_zarr_positions_with_n():
    """Test storing positional embeddings with N index in zarr."""
    from qlty.patch_pairs_2d import extract_patch_pairs_metadata

    tensor = torch.randn(3, 2, 64, 64)
    window = (16, 16)
    num_patches = 4
    delta_range = (6.0, 10.0)

    metadata = extract_patch_pairs_metadata(
        tensor,
        window,
        num_patches,
        delta_range,
        random_seed=42,
    )

    selected_indices = [0, 1, 2, 5]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = str(Path(tmpdir) / "patches.zarr")

        zarr_group = extract_patches_to_zarr(
            tensor,
            metadata,
            selected_indices,
            zarr_path,
            store_positions=True,
            include_n_position=True,
        )

        # Check shapes
        assert zarr_group["positions1"].shape == (4, 3)  # [N, Y, X]
        assert zarr_group["positions2"].shape == (4, 3)  # [N, Y, X]

        # Verify values match metadata
        for i, idx in enumerate(selected_indices):
            assert zarr_group["positions1"][i, 0] == metadata["image_idx"][idx].item()
            assert zarr_group["positions1"][i, 1] == metadata["patch1_y"][idx].item()
            assert zarr_group["positions1"][i, 2] == metadata["patch1_x"][idx].item()


@pytest.mark.skipif(zarr is None, reason="zarr not available")
@pytest.mark.skipif(not HAS_ADVANCED_FEATURES, reason="Advanced features not available")
def test_zarr_patch_pair_dataset_with_positions():
    """Test ZarrPatchPairDataset with positional embeddings."""
    from qlty.patch_pairs_2d import extract_patch_pairs_metadata

    tensor = torch.randn(2, 3, 64, 64)
    window = (16, 16)
    num_patches = 5
    delta_range = (6.0, 10.0)

    metadata = extract_patch_pairs_metadata(
        tensor,
        window,
        num_patches,
        delta_range,
        random_seed=42,
    )

    selected_indices = list(range(len(metadata["image_idx"])))

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = str(Path(tmpdir) / "patches.zarr")

        # Store with positions (Y, X only)
        extract_patches_to_zarr(
            tensor,
            metadata,
            selected_indices,
            zarr_path,
            store_positions=True,
        )

        dataset = ZarrPatchPairDataset(zarr_path)

        # Check that positions are returned
        patch1, patch2, delta, rotation, pos1, pos2 = dataset[0]

        assert patch1.shape == (3, 16, 16)
        assert patch2.shape == (3, 16, 16)
        assert delta.shape == (2,)
        assert pos1.shape == (2,)  # [Y, X]
        assert pos2.shape == (2,)  # [Y, X]
        assert pos1.dtype == torch.int64
        assert pos2.dtype == torch.int64


@pytest.mark.skipif(zarr is None, reason="zarr not available")
@pytest.mark.skipif(not HAS_ADVANCED_FEATURES, reason="Advanced features not available")
def test_zarr_patch_pair_dataset_with_positions_n():
    """Test ZarrPatchPairDataset with positional embeddings including N."""
    from qlty.patch_pairs_2d import extract_patch_pairs_metadata

    tensor = torch.randn(3, 2, 64, 64)
    window = (16, 16)
    num_patches = 4
    delta_range = (6.0, 10.0)

    metadata = extract_patch_pairs_metadata(
        tensor,
        window,
        num_patches,
        delta_range,
        random_seed=42,
    )

    selected_indices = list(range(len(metadata["image_idx"])))

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = str(Path(tmpdir) / "patches.zarr")

        # Store with positions including N
        extract_patches_to_zarr(
            tensor,
            metadata,
            selected_indices,
            zarr_path,
            store_positions=True,
            include_n_position=True,
        )

        dataset = ZarrPatchPairDataset(zarr_path)

        # Check that positions are returned
        patch1, patch2, delta, rotation, pos1, pos2 = dataset[0]

        assert pos1.shape == (3,)  # [N, Y, X]
        assert pos2.shape == (3,)  # [N, Y, X]
        assert pos1[0].item() < tensor.shape[0]  # N index valid


# ============================================================================
# Tests for NCYXQuilt.unstitch() with positional embeddings
# ============================================================================


@pytest.mark.skipif(not HAS_QUILT_FEATURES, reason="Quilt features not available")
def test_ncyx_quilt_unstitch_positions_basic():
    """Test NCYXQuilt.unstitch() with positional embeddings (Y, X only)."""
    tensor = torch.randn(3, 2, 64, 64)
    quilt = qlty2D.NCYXQuilt(
        Y=64,
        X=64,
        window=(16, 16),
        step=(16, 16),
        border=None,
    )

    patches, positions = quilt.unstitch(tensor, return_positions=True)

    # Check shapes
    nY, nX = quilt.get_times()
    expected_patches = 3 * nY * nX
    assert patches.shape[0] == expected_patches
    assert positions.shape == (expected_patches, 2)  # [Y, X]

    # Check that positions are valid
    assert torch.all(positions[:, 0] >= 0)  # Y >= 0
    assert torch.all(positions[:, 1] >= 0)  # X >= 0
    assert torch.all(positions[:, 0] + 16 <= 64)  # Y + window <= Y
    assert torch.all(positions[:, 1] + 16 <= 64)  # X + window <= X


@pytest.mark.skipif(not HAS_QUILT_FEATURES, reason="Quilt features not available")
def test_ncyx_quilt_unstitch_positions_with_n():
    """Test NCYXQuilt.unstitch() with positional embeddings including N."""
    tensor = torch.randn(3, 2, 64, 64)
    quilt = qlty2D.NCYXQuilt(
        Y=64,
        X=64,
        window=(16, 16),
        step=(16, 16),
        border=None,
    )

    patches, positions = quilt.unstitch(
        tensor, return_positions=True, include_n_position=True
    )

    # Check shapes
    nY, nX = quilt.get_times()
    expected_patches = 3 * nY * nX
    assert positions.shape == (expected_patches, 3)  # [N, Y, X]

    # Check that N indices are valid
    assert torch.all(positions[:, 0] >= 0)
    assert torch.all(positions[:, 0] < tensor.shape[0])


@pytest.mark.skipif(not HAS_QUILT_FEATURES, reason="Quilt features not available")
def test_ncyx_quilt_unstitch_data_pair_positions():
    """Test NCYXQuilt.unstitch_data_pair() with positional embeddings."""
    tensor_in = torch.randn(2, 3, 64, 64)
    tensor_out = torch.randn(2, 64, 64)
    quilt = qlty2D.NCYXQuilt(
        Y=64,
        X=64,
        window=(16, 16),
        step=(16, 16),
        border=None,
    )

    inp_patches, out_patches, positions = quilt.unstitch_data_pair(
        tensor_in, tensor_out, return_positions=True
    )

    # Check shapes
    nY, nX = quilt.get_times()
    expected_patches = 2 * nY * nX
    assert inp_patches.shape[0] == expected_patches
    assert out_patches.shape[0] == expected_patches
    assert positions.shape == (expected_patches, 2)  # [Y, X]


# ============================================================================
# Tests for LargeNCYXQuilt.unstitch() with positional embeddings
# ============================================================================


@pytest.mark.skipif(not HAS_QUILT_FEATURES, reason="Quilt features not available")
def test_large_ncyx_quilt_unstitch_positions_basic():
    """Test LargeNCYXQuilt.unstitch() with positional embeddings (Y, X only)."""
    tensor = torch.randn(3, 2, 64, 64)
    with tempfile.TemporaryDirectory() as tmpdir:
        quilt = qlty2DLarge.LargeNCYXQuilt(
            filename=str(Path(tmpdir) / "test"),
            N=3,
            Y=64,
            X=64,
            window=(16, 16),
            step=(16, 16),
            border=None,
        )

        patch, position = quilt.unstitch(tensor, index=0, return_positions=True)

        # Check shapes
        assert patch.shape == (2, 16, 16)
        assert position.shape == (2,)  # [Y, X]

        # Check that position is valid
        assert position[0].item() >= 0  # Y >= 0
        assert position[1].item() >= 0  # X >= 0


@pytest.mark.skipif(not HAS_QUILT_FEATURES, reason="Quilt features not available")
def test_large_ncyx_quilt_unstitch_positions_with_n():
    """Test LargeNCYXQuilt.unstitch() with positional embeddings including N."""
    tensor = torch.randn(3, 2, 64, 64)
    with tempfile.TemporaryDirectory() as tmpdir:
        quilt = qlty2DLarge.LargeNCYXQuilt(
            filename=str(Path(tmpdir) / "test"),
            N=3,
            Y=64,
            X=64,
            window=(16, 16),
            step=(16, 16),
            border=None,
        )

        patch, position = quilt.unstitch(
            tensor, index=0, return_positions=True, include_n_position=True
        )

        # Check shapes
        assert position.shape == (3,)  # [N, Y, X]

        # Check that N index is valid
        assert position[0].item() >= 0
        assert position[0].item() < tensor.shape[0]


@pytest.mark.skipif(not HAS_QUILT_FEATURES, reason="Quilt features not available")
def test_large_ncyx_quilt_unstitch_next_positions():
    """Test LargeNCYXQuilt.unstitch_next() with positional embeddings."""
    tensor = torch.randn(2, 3, 64, 64)
    with tempfile.TemporaryDirectory() as tmpdir:
        quilt = qlty2DLarge.LargeNCYXQuilt(
            filename=str(Path(tmpdir) / "test"),
            N=2,
            Y=64,
            X=64,
            window=(16, 16),
            step=(16, 16),
            border=None,
        )

        idx, patch, position = quilt.unstitch_next(tensor, return_positions=True)

        # Check shapes
        assert patch.shape == (3, 16, 16)
        assert position.shape == (2,)  # [Y, X]
        # idx can be np.int64 from numpy.unravel_index
        assert isinstance(idx, (int, np.integer)) if np else isinstance(idx, int)
        assert int(idx) >= 0
        assert int(idx) < quilt.N_chunks


# ============================================================================
# Integration tests
# ============================================================================


def test_positions_match_patch_locations():
    """Test that positional embeddings correctly match patch extraction locations."""
    tensor = torch.randn(2, 3, 64, 64)
    window = (16, 16)
    num_patches = 3
    delta_range = (6.0, 10.0)

    patches1, patches2, deltas, rotations, positions1, positions2 = extract_patch_pairs(
        tensor,
        window,
        num_patches,
        delta_range,
        return_positions=True,
        random_seed=42,
    )

    # Verify that patches match their positions
    for i in range(len(patches1)):
        y1, x1 = positions1[i, 0].item(), positions1[i, 1].item()
        y2, x2 = positions2[i, 0].item(), positions2[i, 1].item()
        n_idx = i // num_patches

        # Extract patches manually at these positions
        manual_patch1 = tensor[n_idx, :, y1 : y1 + window[0], x1 : x1 + window[1]]
        manual_patch2 = tensor[n_idx, :, y2 : y2 + window[0], x2 : x2 + window[1]]

        # Should match (accounting for rotation)
        if rotations[i].item() == 0:
            assert torch.allclose(patches1[i], manual_patch1)
            assert torch.allclose(patches2[i], manual_patch2)


@pytest.mark.skipif(not HAS_ADVANCED_FEATURES, reason="Advanced features not available")
def test_positions_consistency_metadata_vs_direct():
    """Test that positions from metadata extraction are valid and consistent."""
    from qlty.patch_pairs_2d import extract_patch_pairs_metadata

    tensor = torch.randn(2, 3, 64, 64)
    window = (16, 16)
    num_patches = 5
    delta_range = (6.0, 10.0)
    random_seed = 42

    # Direct extraction
    _, _, _, _, pos1_direct, pos2_direct = extract_patch_pairs(
        tensor,
        window,
        num_patches,
        delta_range,
        return_positions=True,
        random_seed=random_seed,
    )

    # Metadata extraction (use num_workers=1 to ensure reproducibility)
    metadata = extract_patch_pairs_metadata(
        tensor,
        window,
        num_patches,
        delta_range,
        random_seed=random_seed,
        num_workers=1,  # Disable multiprocessing for reproducibility
    )

    selected_indices = list(range(len(metadata["image_idx"])))
    _, _, _, _, pos1_meta, pos2_meta = extract_patches_from_metadata(
        tensor,
        metadata,
        selected_indices,
        return_positions=True,
    )

    # Check that both methods produce valid positions
    # Note: They may not match exactly because metadata extraction uses
    # seed offsets per image (random_seed + image_idx)
    Y, X = tensor.shape[2], tensor.shape[3]

    # Direct extraction positions should be valid
    assert torch.all(pos1_direct[:, 0] >= 0)  # Y >= 0
    assert torch.all(pos1_direct[:, 1] >= 0)  # X >= 0
    assert torch.all(pos1_direct[:, 0] + window[0] <= Y)
    assert torch.all(pos1_direct[:, 1] + window[1] <= X)

    # Metadata extraction positions should be valid
    assert torch.all(pos1_meta[:, 0] >= 0)  # Y >= 0
    assert torch.all(pos1_meta[:, 1] >= 0)  # X >= 0
    assert torch.all(pos1_meta[:, 0] + window[0] <= Y)
    assert torch.all(pos1_meta[:, 1] + window[1] <= X)

    # Both should have same shape
    assert pos1_direct.shape == pos1_meta.shape
    assert pos2_direct.shape == pos2_meta.shape

    # Check that positions match metadata coordinates
    for i, idx in enumerate(selected_indices):
        assert pos1_meta[i, 0].item() == metadata["patch1_y"][idx].item()
        assert pos1_meta[i, 1].item() == metadata["patch1_x"][idx].item()
        assert pos2_meta[i, 0].item() == metadata["patch2_y"][idx].item()
        assert pos2_meta[i, 1].item() == metadata["patch2_x"][idx].item()
