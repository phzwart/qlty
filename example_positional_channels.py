#!/usr/bin/env python
"""
Example demonstrating positional embedding channels in NCYXQuilt.

This example shows how to add positional embedding channels directly to patches
using the NCYXQuilt framework.
"""

import torch

from qlty.qlty2D import NCYXQuilt


def main():
    # Create sample data: 2 images, 3 channels, 128x128
    data = torch.randn(2, 3, 128, 128)

    # Create quilt with 32x32 patches, 16 pixel step
    quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16))

    print("=" * 60)
    print("Example 1: Basic unstitch without positional channels")
    print("=" * 60)
    patches = quilt.unstitch(data)
    print(f"Patches shape: {patches.shape}")  # (M, 3, 32, 32)
    print(f"Number of patches: {patches.shape[0]}")

    print("\n" + "=" * 60)
    print("Example 2: Unstitch with positional channels (absolute, normalized)")
    print("=" * 60)
    patches_with_pos = quilt.unstitch(
        data,
        add_positional_channels=True,
        position_mode="absolute",
        normalize_positions=True,
    )
    print(f"Patches shape: {patches_with_pos.shape}")  # (M, 5, 32, 32)
    print("Channels: 3 original + 2 positional (Y, X)")
    print("Positional channels are normalized to [0, 1]")

    # Check the positional channels
    y_channel = patches_with_pos[0, -2, :, :]  # Second to last channel
    x_channel = patches_with_pos[0, -1, :, :]  # Last channel
    print(f"\nFirst patch Y channel range: [{y_channel.min():.3f}, {y_channel.max():.3f}]")
    print(f"First patch X channel range: [{x_channel.min():.3f}, {x_channel.max():.3f}]")

    print("\n" + "=" * 60)
    print("Example 3: Unstitch with positional channels (relative, normalized)")
    print("=" * 60)
    patches_rel = quilt.unstitch(
        data,
        add_positional_channels=True,
        position_mode="relative",
        normalize_positions=True,
    )
    print(f"Patches shape: {patches_rel.shape}")  # (M, 5, 32, 32)

    y_channel_rel = patches_rel[0, -2, :, :]
    x_channel_rel = patches_rel[0, -1, :, :]
    print(f"\nFirst patch Y channel range (relative): [{y_channel_rel.min():.3f}, {y_channel_rel.max():.3f}]")
    print(f"First patch X channel range (relative): [{x_channel_rel.min():.3f}, {x_channel_rel.max():.3f}]")

    print("\n" + "=" * 60)
    print("Example 4: Unstitch with positional channels (absolute, not normalized)")
    print("=" * 60)
    patches_abs_raw = quilt.unstitch(
        data,
        add_positional_channels=True,
        position_mode="absolute",
        normalize_positions=False,
    )
    print(f"Patches shape: {patches_abs_raw.shape}")  # (M, 5, 32, 32)

    y_channel_raw = patches_abs_raw[0, -2, :, :]
    x_channel_raw = patches_abs_raw[0, -1, :, :]
    print(f"\nFirst patch Y channel range (raw): [{y_channel_raw.min():.0f}, {y_channel_raw.max():.0f}]")
    print(f"First patch X channel range (raw): [{x_channel_raw.min():.0f}, {x_channel_raw.max():.0f}]")

    print("\n" + "=" * 60)
    print("Example 5: Unstitch with both positional channels and return positions")
    print("=" * 60)
    patches_both, positions = quilt.unstitch(
        data,
        return_positions=True,
        add_positional_channels=True,
        position_mode="absolute",
        normalize_positions=True,
    )
    print(f"Patches shape: {patches_both.shape}")  # (M, 5, 32, 32)
    print(f"Positions shape: {positions.shape}")  # (M, 2)
    print("\nFirst few positions:")
    print(positions[:5])

    print("\n" + "=" * 60)
    print("Example 6: unstitch_data_pair with positional channels")
    print("=" * 60)
    input_data = torch.randn(2, 3, 128, 128)
    target_data = torch.randn(2, 128, 128)

    inp_patches, tgt_patches = quilt.unstitch_data_pair(
        input_data,
        target_data,
        add_positional_channels=True,
        position_mode="absolute",
        normalize_positions=True,
    )
    print(f"Input patches shape: {inp_patches.shape}")  # (M, 5, 32, 32)
    print(f"Target patches shape: {tgt_patches.shape}")  # (M, 32, 32)
    print("Note: Positional channels are only added to input patches, not target")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Positional embedding channels can be added to patches using:")
    print("  - add_positional_channels=True")
    print("  - position_mode='absolute' (image coordinates) or 'relative' (patch coordinates)")
    print("  - normalize_positions=True/False (normalize to [0,1] or use raw coordinates)")
    print("\nThis adds 2 channels (Y and X coordinates) to each patch,")
    print("increasing the channel dimension from C to C+2.")


if __name__ == "__main__":
    main()
