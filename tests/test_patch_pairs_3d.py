#!/usr/bin/env python

"""Tests for 3D patch pair extraction functionality."""

import pytest
import torch

from qlty.patch_pairs_3d import (
    _sample_displacement_vector_3d,
    extract_overlapping_pixels_3d,
    extract_patch_pairs_3d,
)


def test_extract_patch_pairs_3d_basic():
    """Test basic 3D patch pair extraction."""
    # Create a simple test tensor: 2 volumes, 2 channels, 32x32x32
    tensor = torch.randn(2, 2, 32, 32, 32)
    window = (8, 8, 8)  # 8x8x8 patches, max_window=8, so delta_range must be in [2, 6]
    num_patches = 3
    delta_range = (3.0, 5.0)  # Valid range for 8x8x8 window

    patches1, patches2, deltas = extract_patch_pairs_3d(
        tensor, window, num_patches, delta_range
    )

    # Check output shapes
    assert patches1.shape == (2 * num_patches, 2, 8, 8, 8)
    assert patches2.shape == (2 * num_patches, 2, 8, 8, 8)
    assert deltas.shape == (2 * num_patches, 3)

    # Check that deltas are floats
    assert deltas.dtype == torch.float32

    # Check that patches are same dtype as input
    assert patches1.dtype == tensor.dtype
    assert patches2.dtype == tensor.dtype


def test_extract_patch_pairs_3d_delta_constraints():
    """Test that 3D delta vectors satisfy Euclidean distance constraints."""
    tensor = torch.randn(1, 1, 64, 64, 64)
    window = (16, 16, 16)  # max_window=16, so delta_range must be in [4, 12]
    num_patches = 10
    delta_range = (6.0, 10.0)  # Valid range

    patches1, patches2, deltas = extract_patch_pairs_3d(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # Check that all delta vectors satisfy the Euclidean distance constraint
    for i in range(deltas.shape[0]):
        dx, dy, dz = deltas[i, 0].item(), deltas[i, 1].item(), deltas[i, 2].item()
        distance = (dx**2 + dy**2 + dz**2) ** 0.5
        assert (
            delta_range[0] <= distance <= delta_range[1]
        ), f"Delta {i}: ({dx}, {dy}, {dz}) has distance {distance}, not in [{delta_range[0]}, {delta_range[1]}]"


def test_extract_patch_pairs_3d_reproducibility():
    """Test that 3D results are reproducible with the same seed."""
    tensor = torch.randn(2, 2, 32, 32, 32)
    window = (8, 8, 8)  # max_window=8, so delta_range must be in [2, 6]
    num_patches = 3
    delta_range = (3.0, 5.0)  # Valid range

    # Extract with same seed twice
    patches1_a, patches2_a, deltas_a = extract_patch_pairs_3d(
        tensor, window, num_patches, delta_range, random_seed=123
    )
    patches1_b, patches2_b, deltas_b = extract_patch_pairs_3d(
        tensor, window, num_patches, delta_range, random_seed=123
    )

    # Results should be identical
    assert torch.allclose(patches1_a, patches1_b)
    assert torch.allclose(patches2_a, patches2_b)
    assert torch.allclose(deltas_a, deltas_b)


def test_extract_patch_pairs_3d_invalid_input_shape():
    """Test that invalid input shapes raise appropriate errors."""
    # Wrong number of dimensions
    tensor_4d = torch.randn(5, 3, 32, 32)
    window = (8, 8, 8)
    num_patches = 5
    delta_range = (4.0, 8.0)

    with pytest.raises(ValueError, match="Input tensor must be 5D"):
        extract_patch_pairs_3d(tensor_4d, window, num_patches, delta_range)


def test_extract_patch_pairs_3d_invalid_delta_range():
    """Test that invalid delta ranges raise appropriate errors."""
    tensor = torch.randn(1, 1, 64, 64, 64)
    window = (16, 16, 16)  # max_window = 16, so window//4 = 4, 3*window//4 = 12
    num_patches = 5

    # Test: low < window//4
    with pytest.raises(ValueError, match="delta_range must satisfy"):
        extract_patch_pairs_3d(tensor, window, num_patches, (2.0, 10.0))

    # Test: high > 3*window//4
    with pytest.raises(ValueError, match="delta_range must satisfy"):
        extract_patch_pairs_3d(tensor, window, num_patches, (6.0, 20.0))

    # Test: low > high
    with pytest.raises(ValueError, match="low.*must be <= high"):
        extract_patch_pairs_3d(tensor, window, num_patches, (10.0, 5.0))


def test_extract_patch_pairs_3d_volume_too_small():
    """Volumes that are too small for the requested window/high delta raise errors."""
    tensor = torch.randn(1, 1, 20, 20, 20)
    window = (16, 16, 16)
    delta_range = (4.0, 8.0)  # max_window//4 = 4, 3*max_window//4 = 12

    with pytest.raises(ValueError, match="Volume dimensions"):
        extract_patch_pairs_3d(tensor, window, num_patches=1, delta_range=delta_range)


def test_extract_patch_pairs_3d_exhausts_invalid_displacements(monkeypatch):
    """If displacement sampling never yields a valid location we raise a ValueError."""
    tensor = torch.randn(1, 1, 32, 32, 32)
    window = (8, 8, 8)
    delta_range = (3.0, 5.0)

    def always_invalid(low, high, generator=None, device=None):
        return 100, 0, 0  # too large to ever fit

    monkeypatch.setattr(
        "qlty.patch_pairs_3d._sample_displacement_vector_3d", always_invalid
    )

    with pytest.raises(ValueError, match="Could not find valid patch locations"):
        extract_patch_pairs_3d(tensor, window, num_patches=1, delta_range=delta_range)


def test_sample_displacement_vector_3d_fallback_scaling(monkeypatch):
    """Fallback path (after max attempts) returns a displacement on CPU without device."""

    def fake_randint(low, high, size, generator=None, device=None):
        device = device or torch.device("cpu")
        shape = size if isinstance(size, tuple) else (size,)
        return torch.zeros(shape, dtype=torch.int64, device=device)

    rand_values = iter([0.0, 0.254, 0.0])  # theta, phi, distance controls

    def fake_rand(size, generator=None, device=None):
        device = device or torch.device("cpu")
        shape = size if isinstance(size, tuple) else (size,)
        value = next(rand_values)
        return torch.full(shape, value, dtype=torch.float32, device=device)

    monkeypatch.setattr("qlty.patch_pairs_3d.torch.randint", fake_randint)
    monkeypatch.setattr("qlty.patch_pairs_3d.torch.rand", fake_rand)

    dx, dy, dz = _sample_displacement_vector_3d(low=3.0, high=5.0)

    # Returned vector should be integral and the fallback path should have been used.
    assert (dx, dy, dz) == (2, 0, 2)


def test_sample_displacement_vector_3d_fallback_with_generator(monkeypatch):
    """Fallback path also works when a torch.Generator is provided."""

    generator = torch.Generator()
    generator.manual_seed(0)

    def fake_randint(low, high, size, generator=None, device=None):
        device = device or torch.device("cpu")
        shape = size if isinstance(size, tuple) else (size,)
        return torch.zeros(shape, dtype=torch.int64, device=device)

    rand_values = iter(
        [0.125, 0.5, 1.0]
    )  # theta -> pi/4, phi -> pi/2, distance -> high

    def fake_rand(size, generator=None, device=None):
        device = device or torch.device("cpu")
        shape = size if isinstance(size, tuple) else (size,)
        value = next(rand_values)
        return torch.full(shape, value, dtype=torch.float32, device=device)

    monkeypatch.setattr("qlty.patch_pairs_3d.torch.randint", fake_randint)
    monkeypatch.setattr("qlty.patch_pairs_3d.torch.rand", fake_rand)

    dx, dy, dz = _sample_displacement_vector_3d(low=4.0, high=5.5, generator=generator)

    assert (dx, dy, dz) == (4, 4, 0)


def test_extract_overlapping_pixels_3d_basic():
    """Test basic 3D overlapping pixel extraction."""
    # Create simple patch pairs
    patches1 = torch.randn(3, 2, 8, 8, 8)  # 3 patch pairs, 2 channels, 8x8x8 patches
    patches2 = torch.randn(3, 2, 8, 8, 8)
    # Deltas: first pair has dx=2, dy=1, dz=1
    #         second pair has dx=-1, dy=-2, dz=0
    #         third pair has dx=0, dy=0, dz=0 (no displacement, full overlap)
    deltas = torch.tensor([[2.0, 1.0, 1.0], [-1.0, -2.0, 0.0], [0.0, 0.0, 0.0]])

    overlapping1, overlapping2 = extract_overlapping_pixels_3d(
        patches1, patches2, deltas
    )

    # Check output shapes
    assert len(overlapping1.shape) == 2
    assert len(overlapping2.shape) == 2
    assert overlapping1.shape == overlapping2.shape
    assert overlapping1.shape[1] == 2  # C channels

    # All should have some overlap (at least the third pair with dx=0,dy=0,dz=0)
    assert overlapping1.shape[0] > 0
    assert overlapping2.shape[0] > 0

    # Check that all values are finite
    assert torch.all(torch.isfinite(overlapping1))
    assert torch.all(torch.isfinite(overlapping2))


def test_extract_overlapping_pixels_3d_no_overlap():
    """Test with 3D patches that have no overlap."""
    patches1 = torch.randn(2, 1, 4, 4, 4)
    patches2 = torch.randn(2, 1, 4, 4, 4)
    # Large displacements that cause no overlap
    deltas = torch.tensor([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]])

    overlapping1, overlapping2 = extract_overlapping_pixels_3d(
        patches1, patches2, deltas
    )

    # Should return empty tensors with correct shape
    assert overlapping1.shape == (0, 1)
    assert overlapping2.shape == (0, 1)
    assert overlapping1.dtype == patches1.dtype
    assert overlapping2.dtype == patches1.dtype
    assert overlapping1.device == patches1.device
    assert overlapping2.device == patches1.device


def test_extract_overlapping_pixels_3d_full_overlap():
    """Test with 3D patches that fully overlap (dx=0, dy=0, dz=0)."""
    patches1 = torch.randn(2, 3, 8, 8, 8)
    patches2 = torch.randn(2, 3, 8, 8, 8)
    deltas = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    overlapping1, overlapping2 = extract_overlapping_pixels_3d(
        patches1, patches2, deltas
    )

    # Should have all pixels from both patches
    assert overlapping1.shape == (2 * 8 * 8 * 8, 3)
    assert overlapping2.shape == (2 * 8 * 8 * 8, 3)
    # Check that values match patches1 and patches2
    assert torch.allclose(
        overlapping1[:512], patches1[0].permute(1, 2, 3, 0).reshape(-1, 3)
    )
    assert torch.allclose(
        overlapping1[512:], patches1[1].permute(1, 2, 3, 0).reshape(-1, 3)
    )
    assert torch.allclose(
        overlapping2[:512], patches2[0].permute(1, 2, 3, 0).reshape(-1, 3)
    )
    assert torch.allclose(
        overlapping2[512:], patches2[1].permute(1, 2, 3, 0).reshape(-1, 3)
    )


def test_extract_overlapping_pixels_3d_invalid_inputs():
    """Test error handling for invalid inputs."""
    patches1 = torch.randn(5, 3, 8, 8, 8)
    patches2 = torch.randn(5, 3, 8, 8, 8)
    deltas = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    # Wrong number of deltas
    with pytest.raises(ValueError, match="Number of deltas"):
        extract_overlapping_pixels_3d(patches1, patches2, deltas)

    # Wrong shape for patches
    patches1_4d = torch.randn(5, 3, 8, 8)
    with pytest.raises(ValueError, match="must be 5D tensors"):
        extract_overlapping_pixels_3d(patches1_4d, patches2, deltas.repeat(5, 1))

    # Mismatched patch shapes
    patches2_wrong = torch.randn(5, 3, 10, 10, 10)
    with pytest.raises(ValueError, match="must have the same shape"):
        extract_overlapping_pixels_3d(patches1, patches2_wrong, deltas.repeat(5, 1))

    # Wrong delta shape
    deltas_wrong = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="must be 2D tensor"):
        extract_overlapping_pixels_3d(patches1, patches2, deltas_wrong)


def test_extract_overlapping_pixels_3d_correspondence():
    """Test that corresponding pixels are at the same index in both tensors."""
    # Create patches with known values to verify correspondence
    patches1 = torch.zeros(2, 1, 8, 8, 8)
    patches2 = torch.zeros(2, 1, 8, 8, 8)

    # Fill patches1 with unique values based on position
    for i in range(2):
        for u in range(8):
            for v in range(8):
                for w in range(8):
                    patches1[i, 0, u, v, w] = i * 1000 + u * 100 + v * 10 + w

    # Fill patches2 with shifted values
    # For pair 0: dx=2, dy=1, dz=1, so patch2[0, u, v, w] should match patch1[0, u+1, v+1, w+2]
    # For pair 1: dx=-1, dy=-1, dz=-1, so patch2[1, u, v, w] should match patch1[1, u-1, v-1, w-1]
    for i in range(2):
        for u in range(8):
            for v in range(8):
                for w in range(8):
                    if i == 0:
                        # dx=2, dy=1, dz=1: patch2[u, v, w] corresponds to patch1[u+1, v+1, w+2]
                        if u + 1 < 8 and v + 1 < 8 and w + 2 < 8:
                            patches2[i, 0, u, v, w] = patches1[
                                i, 0, u + 1, v + 1, w + 2
                            ]
                    else:
                        # dx=-1, dy=-1, dz=-1: patch2[u, v, w] corresponds to patch1[u-1, v-1, w-1]
                        if u - 1 >= 0 and v - 1 >= 0 and w - 1 >= 0:
                            patches2[i, 0, u, v, w] = patches1[
                                i, 0, u - 1, v - 1, w - 1
                            ]

    deltas = torch.tensor([[2.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    overlapping1, overlapping2 = extract_overlapping_pixels_3d(
        patches1, patches2, deltas
    )

    # For corresponding pixels, they should have the same values
    # (since we set them to match)
    assert torch.allclose(overlapping1, overlapping2)


def test_extract_overlapping_pixels_3d_partial_overlap():
    """Test with partial overlap scenarios in 3D."""
    patches1 = torch.randn(4, 2, 10, 10, 10)
    patches2 = torch.randn(4, 2, 10, 10, 10)
    # Various partial overlaps
    deltas = torch.tensor(
        [
            [3.0, 0.0, 0.0],  # X shift only
            [0.0, 4.0, 0.0],  # Y shift only
            [0.0, 0.0, 2.0],  # Z shift only
            [-1.0, -1.0, -1.0],  # Negative diagonal shift
        ]
    )

    overlapping1, overlapping2 = extract_overlapping_pixels_3d(
        patches1, patches2, deltas
    )

    # All should have some overlap
    assert overlapping1.shape[0] > 0
    assert overlapping2.shape[0] > 0
    assert overlapping1.shape == overlapping2.shape
    assert overlapping1.shape[1] == 2
