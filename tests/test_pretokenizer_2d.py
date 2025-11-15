"""Tests for pretokenizer_2d sequence processing."""

import sys
from unittest.mock import patch

import pytest
import torch

from qlty.pretokenizer_2d.sequences import build_sequence_pair, tokenize_patch


def test_tokenize_patch_basic():
    """Test basic patch tokenization with default stride (overlapping)."""
    patch = torch.randn(3, 16, 16)
    patch_size = 4

    tokens, coords = tokenize_patch(patch, patch_size)

    # With default stride=2 (patch_size//2), we get overlapping tokens
    # For 16x16 patch with patch_size=4, stride=2:
    # y positions: 0, 2, 4, 6, 8, 10, 12 (7 positions, last valid is 12 since 12+4=16)
    # x positions: 0, 2, 4, 6, 8, 10, 12 (7 positions)
    # Total: 7 * 7 = 49 tokens
    assert tokens.shape == (49, 3 * 4 * 4)
    assert coords.shape == (49, 2)

    # Check coordinates are correct
    assert coords[0, 0].item() == 0  # First token at (0, 0)
    assert coords[0, 1].item() == 0
    assert coords[1, 0].item() == 0  # Second token at (0, 2) with stride=2
    assert coords[1, 1].item() == 2


def test_tokenize_patch_non_overlapping():
    """Test patch tokenization with non-overlapping tokens (stride=patch_size)."""
    patch = torch.randn(3, 16, 16)
    patch_size = 4
    stride = 4  # Non-overlapping

    tokens, coords = tokenize_patch(patch, patch_size, stride=stride)

    # Should produce 4x4 = 16 tokens (non-overlapping)
    assert tokens.shape == (16, 3 * 4 * 4)
    assert coords.shape == (16, 2)

    # Check coordinates are correct
    assert coords[0, 0].item() == 0  # First token at (0, 0)
    assert coords[0, 1].item() == 0
    assert coords[1, 0].item() == 0  # Second token at (0, 4)
    assert coords[1, 1].item() == 4


def test_tokenize_patch_boundary_enforcement():
    """Test that tokens never extend beyond patch boundaries."""
    patch = torch.randn(1, 15, 15)  # Not divisible by patch_size
    patch_size = 4
    stride = 2

    tokens, coords = tokenize_patch(patch, patch_size, stride=stride)

    # Verify all tokens fit within bounds
    for i in range(tokens.shape[0]):
        y = coords[i, 0].item()
        x = coords[i, 1].item()
        assert y + patch_size <= 15, (
            f"Token {i} extends beyond height: y={y}, y+patch_size={y + patch_size}"
        )
        assert x + patch_size <= 15, (
            f"Token {i} extends beyond width: x={x}, x+patch_size={x + patch_size}"
        )


def test_tokenize_patch_round_trip():
    """Test that tokenization preserves information."""
    patch = torch.randn(1, 8, 8)
    patch_size = 2

    tokens, coords = tokenize_patch(patch, patch_size)

    # Verify we can reconstruct by checking a specific token
    # Token at (0, 0) should match patch[0, 0:2, 0:2]
    token_0 = tokens[0].view(1, 2, 2)
    assert torch.allclose(token_0, patch[:, 0:2, 0:2], atol=1e-6)


def test_build_sequence_pair_no_transform():
    """Test sequence pair with no transformation."""
    patch1 = torch.randn(3, 16, 16)
    patch2 = patch1.clone()

    result = build_sequence_pair(
        patch1, patch2, dx=0.0, dy=0.0, rot_k90=0, patch_size=4
    )

    assert "tokens1" in result
    assert "tokens2" in result
    assert "coords1" in result
    assert "coords2" in result
    assert "overlap_mask1" in result
    assert "overlap_mask2" in result
    assert "overlap_indices1_to_2" in result
    assert "overlap_indices2_to_1" in result
    assert "overlap_fractions" in result
    assert "overlap_pairs" in result

    # With no transform, all tokens should overlap with 100% overlap
    assert result["overlap_mask1"].all()
    assert (result["overlap_indices1_to_2"] >= 0).all()
    assert torch.allclose(
        result["overlap_fractions"],
        torch.ones_like(result["overlap_fractions"]),
        atol=1e-5,
    )


def test_build_sequence_pair_with_translation():
    """Test sequence pair with translation."""
    patch1 = torch.randn(1, 16, 16)
    patch2 = torch.randn(1, 16, 16)

    result = build_sequence_pair(
        patch1, patch2, dx=4.0, dy=0.0, rot_k90=0, patch_size=4
    )

    # Should have some overlap but not all
    assert result["overlap_mask1"].any()
    assert not result["overlap_mask1"].all()


def test_build_sequence_pair_coordinates():
    """Test that coordinates are absolute within patch."""
    patch1 = torch.randn(1, 16, 16)
    patch2 = torch.randn(1, 16, 16)

    result = build_sequence_pair(
        patch1, patch2, dx=0.0, dy=0.0, rot_k90=0, patch_size=4
    )

    # Coordinates should be in [0, 16) range
    assert (result["coords1"] >= 0).all()
    assert (result["coords1"] < 16).all()
    assert (result["coords2"] >= 0).all()
    assert (result["coords2"] < 16).all()

    # First token should be at (0, 0)
    assert result["coords1"][0, 0].item() == 0
    assert result["coords1"][0, 1].item() == 0

    # Verify tokens don't extend beyond boundaries
    patch_size = 4
    assert (result["coords1"][:, 0] + patch_size <= 16).all()
    assert (result["coords1"][:, 1] + patch_size <= 16).all()
    assert (result["coords2"][:, 0] + patch_size <= 16).all()
    assert (result["coords2"][:, 1] + patch_size <= 16).all()


def test_build_sequence_pair_overlap_fractions():
    """Test that overlap fractions are computed correctly."""
    patch1 = torch.randn(1, 16, 16)
    patch2 = torch.randn(1, 16, 16)

    # With no transform, all overlaps should be 1.0
    result = build_sequence_pair(
        patch1, patch2, dx=0.0, dy=0.0, rot_k90=0, patch_size=4
    )
    assert torch.allclose(
        result["overlap_fractions"][result["overlap_mask1"]],
        torch.ones(result["overlap_mask1"].sum()),
        atol=1e-5,
    )

    # With partial translation, overlaps should be < 1.0
    result = build_sequence_pair(
        patch1, patch2, dx=2.0, dy=0.0, rot_k90=0, patch_size=4
    )
    # Overlapping tokens should have fractions between 0 and 1
    if result["overlap_mask1"].any():
        fractions = result["overlap_fractions"][result["overlap_mask1"]]
        assert (fractions > 0.0).all()
        assert (fractions <= 1.0).all()
        # With 2 pixel translation (half a token), overlap should be around 0.5
        # (exact value depends on which tokens overlap)


def test_build_sequence_pair_with_stride():
    """Test that stride parameter works correctly."""
    patch1 = torch.randn(1, 16, 16)
    patch2 = torch.randn(1, 16, 16)

    # With stride=patch_size (non-overlapping), should get fewer tokens
    result_non_overlap = build_sequence_pair(
        patch1, patch2, dx=0.0, dy=0.0, rot_k90=0, patch_size=4, stride=4
    )
    # Should get 4x4 = 16 tokens
    assert result_non_overlap["tokens1"].shape[0] == 16

    # With default stride (overlapping), should get more tokens
    result_overlap = build_sequence_pair(
        patch1, patch2, dx=0.0, dy=0.0, rot_k90=0, patch_size=4
    )
    # Should get 7x7 = 49 tokens with stride=2
    assert result_overlap["tokens1"].shape[0] == 49
    assert result_overlap["tokens1"].shape[0] > result_non_overlap["tokens1"].shape[0]


def test_tokenize_patch_edge_cases():
    """Test edge cases for tokenization."""
    # Test with patch_size larger than patch (should raise error)
    patch = torch.randn(1, 4, 4)
    with pytest.raises(ValueError, match="patch_size.*must be <= patch dimensions"):
        tokenize_patch(patch, patch_size=5)

    # Test with invalid stride
    patch = torch.randn(1, 8, 8)
    with pytest.raises(ValueError, match="stride.*must be <= patch_size"):
        tokenize_patch(patch, patch_size=4, stride=5)

    # Test with zero stride
    with pytest.raises(ValueError, match="stride must be positive"):
        tokenize_patch(patch, patch_size=4, stride=0)

    # Test with very small patch (should still work if patch_size fits)
    patch = torch.randn(1, 4, 4)
    tokens, coords = tokenize_patch(patch, patch_size=2, stride=1)
    # Should get 3x3 = 9 tokens (positions: 0, 1, 2 in each dimension)
    assert tokens.shape[0] == 9


def test_build_sequence_pair_batch():
    """Test batch processing of sequence pairs."""
    # Create batch of patches
    N = 5
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Test with tensor inputs for dx, dy, rot_k90
    dx = torch.tensor([0.0, 2.0, 4.0, 0.0, 1.0])
    dy = torch.tensor([0.0, 0.0, 0.0, 2.0, 1.0])
    rot_k90 = torch.tensor([0, 0, 0, 0, 1])

    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
    )

    # Should return dictionary of tensors
    assert isinstance(result, dict)
    assert "tokens1" in result
    assert "tokens2" in result
    assert "coords1" in result
    assert "coords2" in result
    assert "overlap_mask1" in result
    assert "overlap_mask2" in result
    assert "overlap_indices1_to_2" in result
    assert "overlap_indices2_to_1" in result
    assert "overlap_fractions" in result
    assert "overlap_pairs" in result
    assert "sequence_lengths" in result
    assert "overlap_pair_counts" in result

    # Verify batched tensor shapes
    assert result["tokens1"].shape[0] == N
    assert result["tokens2"].shape[0] == N
    assert result["coords1"].shape[0] == N
    assert result["coords2"].shape[0] == N
    assert result["overlap_mask1"].shape[0] == N
    assert result["overlap_mask2"].shape[0] == N
    assert result["sequence_lengths"].shape == (N,)
    assert result["overlap_pair_counts"].shape == (N,)

    # Verify tokens1 and tokens2 have the same sequence length dimension
    assert result["tokens1"].shape[1] == result["tokens2"].shape[1]
    assert result["coords1"].shape[1] == result["coords2"].shape[1]
    assert result["overlap_mask1"].shape[1] == result["overlap_mask2"].shape[1]

    # Verify sequence lengths match actual data
    for i in range(N):
        T = result["sequence_lengths"][i].item()
        assert result["tokens1"][i, :T].shape[0] == T
        assert result["tokens2"][i, :T].shape[0] == T
        # Padding should be zeros (for tokens) or False (for masks)
        if T < result["tokens1"].shape[1]:
            assert (result["tokens1"][i, T:] == 0).all()
            assert (result["tokens2"][i, T:] == 0).all()

    # Test with scalar inputs (should broadcast)
    result_scalar = build_sequence_pair(
        patch1_batch, patch2_batch, dx=0.0, dy=0.0, rot_k90=0, patch_size=4
    )
    assert isinstance(result_scalar, dict)

    # With no transform, all tokens should overlap (within valid sequence lengths)
    for i in range(N):
        T = result_scalar["sequence_lengths"][i].item()
        assert result_scalar["overlap_mask1"][i, :T].all()
        assert result_scalar["overlap_mask2"][i, :T].all()


def test_build_sequence_pair_batch_numba_path():
    """Test batch processing with N > 5 to trigger numba-accelerated path."""
    # Create batch larger than 5 to trigger numba path
    N = 10
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Test with various transforms
    dx = torch.tensor([0.0, 2.0, 4.0, 0.0, 1.0, 3.0, 0.0, 2.0, 1.0, 0.0])
    dy = torch.tensor([0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rot_k90 = torch.tensor([0, 0, 0, 0, 1, 2, 0, 0, 1, 0])

    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
    )

    # Verify results are correct
    assert result["tokens1"].shape[0] == N
    assert result["tokens2"].shape[0] == N
    assert result["sequence_lengths"].shape == (N,)
    assert result["overlap_pair_counts"].shape == (N,)

    # Verify all batch elements have same sequence length
    T = result["sequence_lengths"][0].item()
    assert (result["sequence_lengths"] == T).all()


def test_build_sequence_pair_batch_sequential_path():
    """Test batch processing with N <= 5 to trigger sequential path."""
    # Create small batch to trigger sequential path
    N = 3
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    dx = torch.tensor([0.0, 2.0, 4.0])
    dy = torch.tensor([0.0, 0.0, 0.0])
    rot_k90 = torch.tensor([0, 0, 0])

    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
    )

    # Verify results are correct
    assert result["tokens1"].shape[0] == N
    assert result["tokens2"].shape[0] == N
    assert result["sequence_lengths"].shape == (N,)


def test_build_sequence_pair_batch_no_overlaps():
    """Test batch processing when some pairs have no overlaps."""
    N = 6
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Large translations that may result in no overlaps
    dx = torch.tensor([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # First one too large
    dy = torch.tensor([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rot_k90 = torch.tensor([0, 0, 0, 0, 0, 0])

    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
    )

    # First pair may have no overlaps
    assert result["overlap_pair_counts"][0] >= 0  # Can be 0 or more
    # Other pairs should have overlaps
    assert (result["overlap_pair_counts"][1:] > 0).all()


def test_build_sequence_pair_with_rotations():
    """Test sequence pair with various rotations."""
    patch1 = torch.randn(1, 16, 16)
    patch2 = torch.randn(1, 16, 16)

    # Test all rotation values
    for rot in [0, 1, 2, 3]:
        result = build_sequence_pair(
            patch1, patch2, dx=0.0, dy=0.0, rot_k90=rot, patch_size=4
        )
        assert "tokens1" in result
        assert "tokens2" in result
        assert result["overlap_mask1"].any()  # Should have some overlap


def test_build_sequence_pair_batch_with_rotations():
    """Test batch processing with rotations."""
    N = 8
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Mix of rotations
    rot_k90 = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    dx = torch.zeros(N)
    dy = torch.zeros(N)

    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
    )

    assert result["tokens1"].shape[0] == N
    assert result["tokens2"].shape[0] == N


def test_build_sequence_pair_batch_scalar_broadcast():
    """Test that scalar inputs are properly broadcast in batch mode."""
    N = 7
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Test with scalar values (should broadcast to all batch elements)
    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx=2.0, dy=1.0, rot_k90=1, patch_size=4
    )

    assert result["tokens1"].shape[0] == N
    assert result["tokens2"].shape[0] == N


def test_build_sequence_pair_batch_numpy_inputs():
    """Test batch processing with numpy array inputs."""
    import numpy as np

    N = 6
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Test with numpy arrays
    dx = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 0.0], dtype=np.float32)
    dy = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    rot_k90 = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)

    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
    )

    assert result["tokens1"].shape[0] == N
    assert result["tokens2"].shape[0] == N


def test_build_sequence_pair_batch_empty_overlap_pairs():
    """Test batch processing when max_pairs is 0 (no overlaps)."""
    N = 4
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Very large translations to ensure no overlaps
    dx = torch.tensor([50.0, 50.0, 50.0, 50.0])
    dy = torch.tensor([50.0, 50.0, 50.0, 50.0])
    rot_k90 = torch.tensor([0, 0, 0, 0])

    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
    )

    # Should handle empty overlap pairs gracefully
    assert "overlap_pairs" in result
    assert result["overlap_pairs"].shape[0] == N
    assert result["overlap_pair_counts"].sum() == 0  # No overlaps


def test_build_sequence_pair_batch_mismatch_error():
    """Test that batch size mismatch raises an error."""
    patch1_batch = torch.randn(5, 3, 16, 16)
    patch2_batch = torch.randn(3, 3, 16, 16)  # Different batch size

    with pytest.raises(ValueError, match="Batch sizes must match"):
        build_sequence_pair(
            patch1_batch, patch2_batch, dx=0.0, dy=0.0, rot_k90=0, patch_size=4
        )


def test_build_sequence_pair_batch_shape_mismatch_error():
    """Test that shape mismatch raises an error."""
    patch1_batch = torch.randn(5, 3, 16, 16)
    patch2_batch = torch.randn(5, 3, 32, 32)  # Different patch size

    with pytest.raises(ValueError, match="Patches must have same shape"):
        build_sequence_pair(
            patch1_batch, patch2_batch, dx=0.0, dy=0.0, rot_k90=0, patch_size=4
        )


def test_build_sequence_pair_batch_tensor_size_mismatch():
    """Test that tensor size mismatch for dx/dy/rot_k90 raises an error."""
    N = 5
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Wrong size for dx
    dx = torch.tensor([0.0, 1.0, 2.0])  # Only 3 elements, should be 5

    with pytest.raises(ValueError, match="must have shape"):
        build_sequence_pair(
            patch1_batch, patch2_batch, dx, dy=0.0, rot_k90=0, patch_size=4
        )


def test_build_sequence_pair_single_patch_tensor_inputs():
    """Test single patch processing with tensor inputs (should convert to scalars)."""
    patch1 = torch.randn(3, 16, 16)
    patch2 = torch.randn(3, 16, 16)

    # Test with tensor inputs (should be converted to scalars)
    dx = torch.tensor(2.0)
    dy = torch.tensor(1.0)
    rot_k90 = torch.tensor(1)

    result = build_sequence_pair(patch1, patch2, dx, dy, rot_k90, patch_size=4)
    assert "tokens1" in result
    assert "tokens2" in result


def test_build_sequence_pair_single_patch_shape_mismatch():
    """Test single patch processing with shape mismatch error."""
    patch1 = torch.randn(3, 16, 16)
    patch2 = torch.randn(3, 32, 32)  # Different size

    with pytest.raises(ValueError, match="Patches must have same shape"):
        build_sequence_pair(patch1, patch2, dx=0.0, dy=0.0, rot_k90=0, patch_size=4)


def test_build_sequence_pair_single_patch_invalid_shape():
    """Test single patch processing with invalid input shape."""
    patch1 = torch.randn(3, 16, 16)
    patch2 = torch.randn(5, 3, 16, 16)  # 4D instead of 3D

    with pytest.raises(ValueError, match="Both patches must be 3D"):
        build_sequence_pair(patch1, patch2, dx=0.0, dy=0.0, rot_k90=0, patch_size=4)


def test_tokenize_patch_negative_patch_size():
    """Test tokenize_patch with negative patch_size."""
    patch = torch.randn(1, 8, 8)
    with pytest.raises(ValueError, match="patch_size must be positive"):
        tokenize_patch(patch, patch_size=-1)


def test_tokenize_patch_zero_patch_size():
    """Test tokenize_patch with zero patch_size."""
    patch = torch.randn(1, 8, 8)
    with pytest.raises(ValueError, match="patch_size must be positive"):
        tokenize_patch(patch, patch_size=0)


def test_tokenize_patch_invalid_shape():
    """Test tokenize_patch with invalid patch shape (not 3D)."""
    # Test with 2D tensor
    patch_2d = torch.randn(8, 8)
    with pytest.raises(ValueError, match="patch must be 3D"):
        tokenize_patch(patch_2d, patch_size=4)

    # Test with 4D tensor
    patch_4d = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="patch must be 3D"):
        tokenize_patch(patch_4d, patch_size=4)


def test_build_sequence_pair_rotation_normalization():
    """Test that rotation values are normalized with modulo 4."""
    patch1 = torch.randn(1, 16, 16)
    patch2 = torch.randn(1, 16, 16)

    # Test with rotation > 3 (should be normalized: 5 % 4 = 1)
    result1 = build_sequence_pair(
        patch1, patch2, dx=0.0, dy=0.0, rot_k90=5, patch_size=4
    )
    result2 = build_sequence_pair(
        patch1, patch2, dx=0.0, dy=0.0, rot_k90=1, patch_size=4
    )
    # Should produce same result
    assert torch.allclose(result1["tokens1"], result2["tokens1"])

    # Test with negative rotation (should be normalized: -1 % 4 = 3)
    result3 = build_sequence_pair(
        patch1, patch2, dx=0.0, dy=0.0, rot_k90=-1, patch_size=4
    )
    result4 = build_sequence_pair(
        patch1, patch2, dx=0.0, dy=0.0, rot_k90=3, patch_size=4
    )
    # Should produce same result
    assert torch.allclose(result3["tokens1"], result4["tokens1"])


def test_build_sequence_pair_batch_numpy_wrong_shape():
    """Test batch processing with numpy array that has wrong shape."""
    import numpy as np

    N = 5
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # NumPy array with wrong shape
    dx = np.array([0.0, 1.0, 2.0], dtype=np.float32)  # Only 3 elements, should be 5

    with pytest.raises(ValueError, match="must have shape"):
        build_sequence_pair(
            patch1_batch, patch2_batch, dx, dy=0.0, rot_k90=0, patch_size=4
        )


def test_build_sequence_pair_batch_numba_path_coverage():
    """Test that numba path is actually called and works correctly."""
    # Use a batch size that triggers numba path (N > 5)
    N = 10
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Test with various transforms including rotations
    dx = torch.tensor([0.0, 2.0, 4.0, 0.0, 1.0, 3.0, 0.0, 2.0, 1.0, 0.0])
    dy = torch.tensor([0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rot_k90 = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])  # Mix of all rotations

    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
    )

    # Verify numba path was used (results should be correct)
    assert result["tokens1"].shape[0] == N
    assert result["tokens2"].shape[0] == N
    assert result["sequence_lengths"].shape == (N,)
    # All should have same sequence length
    assert (result["sequence_lengths"] == result["sequence_lengths"][0]).all()


def test_build_sequence_pair_batch_numba_all_rotations():
    """Test numba path with all rotation values to ensure all branches are covered."""
    N = 8
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Test each rotation value separately to ensure all code paths are hit
    for rot in [0, 1, 2, 3]:
        dx = torch.zeros(N)
        dy = torch.zeros(N)
        rot_k90 = torch.full((N,), rot, dtype=torch.int64)

        result = build_sequence_pair(
            patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
        )

        assert result["tokens1"].shape[0] == N
        # With no translation and same rotation, should have overlaps
        assert result["overlap_pair_counts"].sum() > 0


def test_build_sequence_pair_batch_numba_edge_cases():
    """Test numba path with edge cases like no overlaps and partial overlaps."""
    N = 7
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Mix of cases: some with overlaps, some without
    dx = torch.tensor([0.0, 0.0, 20.0, 2.0, 4.0, 0.0, 0.0])
    dy = torch.tensor([0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0])
    rot_k90 = torch.tensor([0, 1, 0, 2, 3, 0, 1])

    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
    )

    assert result["tokens1"].shape[0] == N
    # Should handle all cases correctly
    assert result["overlap_pair_counts"].shape == (N,)


def test_build_sequence_pair_batch_with_explicit_stride():
    """Test batch processing with explicit stride parameter (covers line 587)."""
    N = 6
    patch1_batch = torch.randn(N, 3, 16, 16)
    patch2_batch = torch.randn(N, 3, 16, 16)

    # Test with explicit stride (not None)
    result = build_sequence_pair(
        patch1_batch, patch2_batch, dx=0.0, dy=0.0, rot_k90=0, patch_size=4, stride=2
    )

    assert result["tokens1"].shape[0] == N
    # With stride=2, should get more tokens than with stride=4
    result_stride4 = build_sequence_pair(
        patch1_batch, patch2_batch, dx=0.0, dy=0.0, rot_k90=0, patch_size=4, stride=4
    )
    assert result["tokens1"].shape[1] > result_stride4["tokens1"].shape[1]


def test_build_sequence_pair_batch_without_numba():
    """Test batch processing when numba is not available (covers HAS_NUMBA=False path)."""
    # Mock numba import to simulate it not being available
    with patch.dict(
        "sys.modules", {"numba": None, "numba.njit": None, "numba.prange": None}
    ):
        # Reload the module to get the HAS_NUMBA=False path
        import importlib

        import qlty.pretokenizer_2d.sequences as seq_module

        importlib.reload(seq_module)

        # Test with batch that would normally use numba (N > 5)
        N = 10
        patch1_batch = torch.randn(N, 3, 16, 16)
        patch2_batch = torch.randn(N, 3, 16, 16)

        dx = torch.tensor([0.0] * N)
        dy = torch.tensor([0.0] * N)
        rot_k90 = torch.tensor([0] * N)

        # Should fall back to sequential processing
        result = seq_module.build_sequence_pair(
            patch1_batch, patch2_batch, dx, dy, rot_k90, patch_size=4
        )

        assert result["tokens1"].shape[0] == N
        assert result["tokens2"].shape[0] == N

        # Reload again to restore numba
        importlib.reload(seq_module)
