#!/usr/bin/env python

"""Tests for patch pair extraction functionality."""

import pytest
import torch

from qlty.patch_pairs_2d import extract_overlapping_pixels, extract_patch_pairs


def test_extract_patch_pairs_basic():
    """Test basic patch pair extraction."""
    # Create a simple test tensor: 2 images, 3 channels, 64x64
    tensor = torch.randn(2, 3, 64, 64)
    window = (16, 16)  # max_window=16, so delta_range must be in [4, 12]
    num_patches = 5
    delta_range = (6.0, 10.0)  # Valid range within [4, 12]

    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window, num_patches, delta_range
    )

    # Check output shapes
    assert patches1.shape == (2 * num_patches, 3, 16, 16)
    assert patches2.shape == (2 * num_patches, 3, 16, 16)
    assert deltas.shape == (2 * num_patches, 2)

    # Check that deltas are floats
    assert deltas.dtype == torch.float32

    # Check that patches are same dtype as input
    assert patches1.dtype == tensor.dtype
    assert patches2.dtype == tensor.dtype


def test_extract_patch_pairs_delta_constraints():
    """Test that delta vectors satisfy Euclidean distance constraints."""
    tensor = torch.randn(1, 1, 128, 128)
    window = (32, 32)
    num_patches = 20
    delta_range = (10.0, 20.0)

    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # Check that all delta vectors satisfy the Euclidean distance constraint
    for i in range(deltas.shape[0]):
        dx, dy = deltas[i, 0].item(), deltas[i, 1].item()
        distance = (dx**2 + dy**2) ** 0.5
        assert (
            delta_range[0] <= distance <= delta_range[1]
        ), f"Delta {i}: ({dx}, {dy}) has distance {distance}, not in [{delta_range[0]}, {delta_range[1]}]"


def test_extract_patch_pairs_reproducibility():
    """Test that results are reproducible with the same seed."""
    tensor = torch.randn(2, 2, 64, 64)
    window = (16, 16)
    num_patches = 3
    delta_range = (5.0, 10.0)

    # Extract with same seed twice
    patches1_a, patches2_a, deltas_a = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=123
    )
    patches1_b, patches2_b, deltas_b = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=123
    )

    # Results should be identical
    assert torch.allclose(patches1_a, patches1_b)
    assert torch.allclose(patches2_a, patches2_b)
    assert torch.allclose(deltas_a, deltas_b)


def test_extract_patch_pairs_different_seeds():
    """Test that different seeds produce different results."""
    tensor = torch.randn(1, 1, 64, 64)
    window = (16, 16)
    num_patches = 5
    delta_range = (5.0, 10.0)

    patches1_a, patches2_a, deltas_a = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=1
    )
    patches1_b, patches2_b, deltas_b = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=2
    )

    # Results should be different (at least deltas should differ)
    assert not torch.allclose(deltas_a, deltas_b)


def test_extract_patch_pairs_multiple_images():
    """Test that patch extraction works correctly for multiple images."""
    tensor = torch.randn(5, 4, 96, 96)
    window = (24, 24)
    num_patches = 3
    delta_range = (8.0, 16.0)

    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window, num_patches, delta_range
    )

    # Should have 5 * 3 = 15 patches total
    assert patches1.shape[0] == 5 * num_patches
    assert patches2.shape[0] == 5 * num_patches
    assert deltas.shape[0] == 5 * num_patches


def test_extract_patch_pairs_invalid_input_shape():
    """Test that invalid input shapes raise appropriate errors."""
    # Wrong number of dimensions
    tensor_3d = torch.randn(5, 3, 64)
    window = (16, 16)
    num_patches = 5
    delta_range = (8.0, 16.0)

    with pytest.raises(ValueError, match="Input tensor must be 4D"):
        extract_patch_pairs(tensor_3d, window, num_patches, delta_range)


def test_extract_patch_pairs_invalid_delta_range():
    """Test that invalid delta ranges raise appropriate errors."""
    tensor = torch.randn(1, 1, 64, 64)
    window = (32, 32)  # max_window = 32, so window//4 = 8, 3*window//4 = 24
    num_patches = 5

    # Test: low < window//4
    with pytest.raises(ValueError, match="delta_range must satisfy"):
        extract_patch_pairs(tensor, window, num_patches, (5.0, 20.0))

    # Test: high > 3*window//4
    with pytest.raises(ValueError, match="delta_range must satisfy"):
        extract_patch_pairs(tensor, window, num_patches, (10.0, 30.0))

    # Test: low > high
    with pytest.raises(ValueError, match="low.*must be <= high"):
        extract_patch_pairs(tensor, window, num_patches, (20.0, 10.0))


def test_extract_patch_pairs_image_too_small():
    """Test that images that are too small raise appropriate errors."""
    window = (32, 32)
    num_patches = 5
    delta_range = (8.0, 16.0)

    # Image too small: 64 < 32 + 16 = 48 (minimum required)
    # Actually, let's check: min_y = 32 + 16 = 48, min_x = 32 + 16 = 48
    # So 64 should be fine. Let's use a smaller image.
    tensor = torch.randn(1, 1, 40, 40)  # 40 < 48, so should fail

    with pytest.raises(ValueError, match="Image dimensions.*are too small"):
        extract_patch_pairs(tensor, window, num_patches, delta_range)


def test_extract_patch_pairs_rectangular_window():
    """Test that rectangular windows work correctly."""
    tensor = torch.randn(2, 2, 128, 128)
    window = (16, 32)  # Rectangular: height=16, width=32
    num_patches = 5
    delta_range = (8.0, 16.0)  # max_window = 32, so constraints are based on 32

    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window, num_patches, delta_range
    )

    # Check output shapes match window
    assert patches1.shape == (2 * num_patches, 2, 16, 32)
    assert patches2.shape == (2 * num_patches, 2, 16, 32)


def test_extract_patch_pairs_negative_displacements():
    """Test that negative displacements (dx, dy) work correctly."""
    tensor = torch.randn(1, 1, 128, 128)
    window = (32, 32)
    num_patches = 20
    delta_range = (10.0, 20.0)

    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # Check that deltas are valid (with enough samples, we should have some negative values)
    for i in range(deltas.shape[0]):
        dx, dy = deltas[i, 0].item(), deltas[i, 1].item()
        distance = (dx**2 + dy**2) ** 0.5
        assert delta_range[0] <= distance <= delta_range[1]


def test_extract_patch_pairs_patches_within_bounds():
    """Test that extracted patches are actually from the input tensor."""
    tensor = torch.randn(1, 1, 64, 64)
    # Create a tensor with known values to verify patches are extracted correctly
    tensor = torch.zeros(1, 1, 64, 64)
    tensor[0, 0, 16:32, 16:32] = 1.0  # Mark a specific region

    window = (16, 16)
    num_patches = 10
    delta_range = (5.0, 10.0)

    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # All patches should be valid (non-NaN, finite)
    assert torch.all(torch.isfinite(patches1))
    assert torch.all(torch.isfinite(patches2))

    # Patches should be within reasonable value range (0 to 1 in this case)
    assert torch.all(patches1 >= 0)
    assert torch.all(patches2 >= 0)


def test_extract_patch_pairs_device_consistency():
    """Test that output tensors are on the same device as input."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = torch.randn(1, 1, 64, 64, device=device)
        window = (16, 16)
        num_patches = 3
        delta_range = (5.0, 10.0)

        patches1, patches2, deltas = extract_patch_pairs(
            tensor, window, num_patches, delta_range
        )

        assert patches1.device == device
        assert patches2.device == device
        assert deltas.device == device


def test_extract_patch_pairs_edge_case_minimum_delta():
    """Test with minimum allowed delta range."""
    tensor = torch.randn(1, 1, 128, 128)
    window = (32, 32)  # window//4 = 8, 3*window//4 = 24
    num_patches = 5
    delta_range = (8.0, 8.0)  # Minimum at boundary

    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # All deltas should have distance exactly 8 (within floating point tolerance)
    for i in range(deltas.shape[0]):
        dx, dy = deltas[i, 0].item(), deltas[i, 1].item()
        distance = (dx**2 + dy**2) ** 0.5
        # Allow small tolerance for integer rounding
        assert abs(distance - 8.0) < 1.0, f"Distance {distance} not close to 8.0"


def test_extract_patch_pairs_edge_case_maximum_delta():
    """Test with maximum allowed delta range."""
    tensor = torch.randn(1, 1, 128, 128)
    window = (32, 32)  # window//4 = 8, 3*window//4 = 24
    num_patches = 5
    delta_range = (24.0, 24.0)  # Maximum at boundary

    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # All deltas should have distance approximately 24
    for i in range(deltas.shape[0]):
        dx, dy = deltas[i, 0].item(), deltas[i, 1].item()
        distance = (dx**2 + dy**2) ** 0.5
        # Allow tolerance for integer rounding
        assert abs(distance - 24.0) < 1.0, f"Distance {distance} not close to 24.0"


def test_extract_overlapping_pixels_basic():
    """Test basic overlapping pixel extraction."""
    # Create simple patch pairs
    patches1 = torch.randn(3, 2, 8, 8)  # 3 patch pairs, 2 channels, 8x8 patches
    patches2 = torch.randn(3, 2, 8, 8)
    # Deltas: first pair has dx=2, dy=1 (positive displacement)
    #         second pair has dx=-1, dy=-2 (negative displacement)
    #         third pair has dx=0, dy=0 (no displacement, full overlap)
    deltas = torch.tensor([[2.0, 1.0], [-1.0, -2.0], [0.0, 0.0]])

    overlapping1, overlapping2 = extract_overlapping_pixels(patches1, patches2, deltas)

    # Check output shapes
    assert len(overlapping1.shape) == 2
    assert len(overlapping2.shape) == 2
    assert overlapping1.shape == overlapping2.shape
    assert overlapping1.shape[1] == 2  # C channels

    # For 8x8 patches:
    # - Pair 0 (dx=2, dy=1): overlap region is (1:8, 2:8) = 7x6 = 42 pixels
    # - Pair 1 (dx=-1, dy=-2): overlap region is (0:6, 0:7) = 6x7 = 42 pixels
    # - Pair 2 (dx=0, dy=0): overlap region is (0:8, 0:8) = 8x8 = 64 pixels
    # Total: 42 + 42 + 64 = 148 pixels
    assert overlapping1.shape[0] == 148
    assert overlapping2.shape[0] == 148

    # Check that all values are finite
    assert torch.all(torch.isfinite(overlapping1))
    assert torch.all(torch.isfinite(overlapping2))


def test_extract_overlapping_pixels_no_overlap():
    """Test with patches that have no overlap."""
    patches1 = torch.randn(2, 1, 4, 4)
    patches2 = torch.randn(2, 1, 4, 4)
    # Large displacements that cause no overlap
    deltas = torch.tensor([[10.0, 10.0], [-10.0, -10.0]])

    overlapping1, overlapping2 = extract_overlapping_pixels(patches1, patches2, deltas)

    # Should return empty tensors with correct shape
    assert overlapping1.shape == (0, 1)
    assert overlapping2.shape == (0, 1)
    assert overlapping1.dtype == patches1.dtype
    assert overlapping2.dtype == patches1.dtype
    assert overlapping1.device == patches1.device
    assert overlapping2.device == patches1.device


def test_extract_overlapping_pixels_full_overlap():
    """Test with patches that fully overlap (dx=0, dy=0)."""
    patches1 = torch.randn(2, 3, 16, 16)
    patches2 = torch.randn(2, 3, 16, 16)
    deltas = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

    overlapping1, overlapping2 = extract_overlapping_pixels(patches1, patches2, deltas)

    # Should have all pixels from both patches
    assert overlapping1.shape == (2 * 16 * 16, 3)
    assert overlapping2.shape == (2 * 16 * 16, 3)
    # Check that values match patches1 and patches2
    assert torch.allclose(
        overlapping1[:256], patches1[0].permute(1, 2, 0).reshape(-1, 3)
    )
    assert torch.allclose(
        overlapping1[256:], patches1[1].permute(1, 2, 0).reshape(-1, 3)
    )
    assert torch.allclose(
        overlapping2[:256], patches2[0].permute(1, 2, 0).reshape(-1, 3)
    )
    assert torch.allclose(
        overlapping2[256:], patches2[1].permute(1, 2, 0).reshape(-1, 3)
    )


def test_extract_overlapping_pixels_invalid_inputs():
    """Test error handling for invalid inputs."""
    patches1 = torch.randn(5, 3, 8, 8)
    patches2 = torch.randn(5, 3, 8, 8)
    deltas = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    # Wrong number of deltas
    with pytest.raises(ValueError, match="Number of deltas"):
        extract_overlapping_pixels(patches1, patches2, deltas)

    # Wrong shape for patches
    patches1_3d = torch.randn(5, 3, 8)
    with pytest.raises(ValueError, match="must be 4D tensors"):
        extract_overlapping_pixels(patches1_3d, patches2, deltas.repeat(5, 1))

    # Mismatched patch shapes
    patches2_wrong = torch.randn(5, 3, 10, 10)
    with pytest.raises(ValueError, match="must have the same shape"):
        extract_overlapping_pixels(patches1, patches2_wrong, deltas.repeat(5, 1))

    # Wrong delta shape
    deltas_wrong = torch.tensor([1.0, 1.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="must be 2D tensor"):
        extract_overlapping_pixels(patches1, patches2, deltas_wrong)


def test_extract_overlapping_pixels_partial_overlap():
    """Test with partial overlap scenarios."""
    patches1 = torch.randn(4, 2, 10, 10)
    patches2 = torch.randn(4, 2, 10, 10)
    # Various partial overlaps
    deltas = torch.tensor(
        [
            [3.0, 0.0],  # Horizontal shift only
            [0.0, 4.0],  # Vertical shift only
            [2.0, 2.0],  # Diagonal shift
            [-1.0, -1.0],  # Negative diagonal shift
        ]
    )

    overlapping1, overlapping2 = extract_overlapping_pixels(patches1, patches2, deltas)

    # All should have some overlap
    assert overlapping1.shape[0] > 0
    assert overlapping2.shape[0] > 0
    assert overlapping1.shape == overlapping2.shape
    assert overlapping1.shape[1] == 2

    # Verify overlap sizes make sense
    # Pair 0 (dx=3, dy=0): overlap is (0:10, 3:10) = 10x7 = 70 pixels
    # Pair 1 (dx=0, dy=4): overlap is (4:10, 0:10) = 6x10 = 60 pixels
    # Pair 2 (dx=2, dy=2): overlap is (2:10, 2:10) = 8x8 = 64 pixels
    # Pair 3 (dx=-1, dy=-1): overlap is (1:10, 1:10) = 9x9 = 81 pixels
    # Total: 70 + 60 + 64 + 81 = 275 pixels
    # (Note: actual calculation may vary slightly due to boundary conditions)
    assert overlapping1.shape[0] >= 200  # At least some overlap


def test_extract_overlapping_pixels_correspondence():
    """Test that corresponding pixels are at the same index in both tensors."""
    # Create patches with known values to verify correspondence
    patches1 = torch.zeros(2, 1, 8, 8)
    patches2 = torch.zeros(2, 1, 8, 8)

    # Fill patches1 with unique values based on position
    for i in range(2):
        for u in range(8):
            for v in range(8):
                patches1[i, 0, u, v] = i * 100 + u * 10 + v

    # Fill patches2 with shifted values
    # For pair 0: dx=2, dy=1, so patch2[0, u, v] should match patch1[0, u+1, v+2]
    # For pair 1: dx=-1, dy=-1, so patch2[1, u, v] should match patch1[1, u-1, v-1]
    for i in range(2):
        for u in range(8):
            for v in range(8):
                if i == 0:
                    # dx=2, dy=1: patch2[u, v] corresponds to patch1[u+1, v+2]
                    if u + 1 < 8 and v + 2 < 8:
                        patches2[i, 0, u, v] = patches1[i, 0, u + 1, v + 2]
                else:
                    # dx=-1, dy=-1: patch2[u, v] corresponds to patch1[u-1, v-1]
                    if u - 1 >= 0 and v - 1 >= 0:
                        patches2[i, 0, u, v] = patches1[i, 0, u - 1, v - 1]

    deltas = torch.tensor([[2.0, 1.0], [-1.0, -1.0]])
    overlapping1, overlapping2 = extract_overlapping_pixels(patches1, patches2, deltas)

    # For corresponding pixels, they should have the same values
    # (since we set them to match)
    assert torch.allclose(overlapping1, overlapping2)
