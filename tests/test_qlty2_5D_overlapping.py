"""
Tests for 2.5D Quilt extract_overlapping_pixels integration.
"""

import pytest
import torch

from qlty.qlty2_5D import NCZYX25DQuilt


def test_extract_overlapping_pixels():
    """Test extract_overlapping_pixels integration."""
    data = torch.randn(2, 1, 10, 50, 50)

    spec = {"identity": [0]}
    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[5],
    )

    overlapping1, overlapping2 = quilt.extract_overlapping_pixels(
        window=(16, 16),
        num_patches=10,
        delta_range=(4.0, 8.0),
        random_seed=42,
    )

    # Check shapes - should be (K, C') where K is number of overlapping pixels
    assert len(overlapping1.shape) == 2
    assert len(overlapping2.shape) == 2
    assert overlapping1.shape == overlapping2.shape
    assert overlapping1.shape[1] == 1  # 1 channel from spec

    # Check that we got some overlapping pixels
    assert overlapping1.shape[0] > 0
    assert overlapping2.shape[0] > 0


def test_extract_overlapping_pixels_requires_2d_mode():
    """Test that extract_overlapping_pixels requires 2d mode."""
    data = torch.randn(1, 1, 10, 50, 50)

    spec = {"identity": [0]}
    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="3d",  # Wrong mode
    )

    with pytest.raises(ValueError, match="accumulation_mode='2d'"):
        quilt.extract_overlapping_pixels(
            window=(16, 16),
            num_patches=10,
            delta_range=(4.0, 8.0),
        )


def test_extract_overlapping_pixels_multiple_channels():
    """Test extract_overlapping_pixels with multiple channels."""
    data = torch.randn(1, 1, 10, 50, 50)

    spec = {
        "identity": [-1, 0, 1],  # 3 channels
        "mean": [[-2, -3]],  # 1 channel
    }
    quilt = NCZYX25DQuilt(
        data_source=data,
        channel_spec=spec,
        accumulation_mode="2d",
        z_slices=[5],
    )

    overlapping1, overlapping2 = quilt.extract_overlapping_pixels(
        window=(16, 16),
        num_patches=10,
        delta_range=(4.0, 8.0),
        random_seed=42,
    )

    # Should have 4 channels (3 direct + 1 mean)
    assert overlapping1.shape[1] == 4
    assert overlapping2.shape[1] == 4


if __name__ == "__main__":
    test_extract_overlapping_pixels()

    test_extract_overlapping_pixels_requires_2d_mode()

    test_extract_overlapping_pixels_multiple_channels()
