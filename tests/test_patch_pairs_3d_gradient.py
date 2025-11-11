#!/usr/bin/env python

"""Test gradient flow and kernel optimization with 3D patch pairs."""

import pytest
import torch
import torch.nn as nn

from qlty.patch_pairs_3d import extract_overlapping_pixels_3d, extract_patch_pairs_3d


def test_kernel_optimization_3d_with_overlapping_pixels():
    """
    Test that optimizing one kernel to match a fixed kernel works using L1 loss
    on overlapping pixels from 3D patch pairs.
    """
    torch.manual_seed(789)

    # Create input tensor: (5, 1, 32, 32, 32)
    input_tensor = torch.randn(5, 1, 32, 32, 32)

    # Extract patch pairs with 9x9x9 window
    window = (9, 9, 9)  # max_window=9, so delta_range must be in [2, 6]
    num_patches = 3  # 3 patch pairs per volume = 15 total pairs
    delta_range = (3.0, 5.0)  # Valid range

    patches1, patches2, deltas = extract_patch_pairs_3d(
        input_tensor, window, num_patches, delta_range, random_seed=789
    )

    # Detach patches for optimization
    patches1 = patches1.detach()
    patches2 = patches2.detach()

    # Verify patch shapes
    assert patches1.shape == (15, 1, 9, 9, 9)
    assert patches2.shape == (15, 1, 9, 9, 9)

    # Create two Conv3D layers with reflection padding
    kernel_size = 3
    padding = kernel_size // 2

    # Fixed kernel (target)
    fixed_conv = nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode="reflect",
        bias=False,
    )

    # Trainable kernel (to be optimized)
    trainable_conv = nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode="reflect",
        bias=False,
    )

    # Initialize trainable kernel with different values
    with torch.no_grad():
        trainable_conv.weight.data = torch.randn_like(trainable_conv.weight.data) * 0.5

    # Store the fixed kernel weights for comparison
    fixed_kernel = fixed_conv.weight.data.clone()

    # Freeze the fixed kernel
    for param in fixed_conv.parameters():
        param.requires_grad = False

    # Set up optimizer for trainable kernel
    optimizer = torch.optim.Adam(trainable_conv.parameters(), lr=0.05)

    # Training loop
    num_iterations = 200
    losses = []
    kernel_diffs = []

    for _ in range(num_iterations):
        optimizer.zero_grad()

        # Apply conv layers to both patch stacks
        output1 = fixed_conv(patches1)  # Shape: (15, 1, 9, 9, 9)
        output2 = trainable_conv(patches2)  # Shape: (15, 1, 9, 9, 9)

        # Extract overlapping pixels
        overlapping1, overlapping2 = extract_overlapping_pixels_3d(
            output1, output2, deltas
        )

        # Compute L1 loss on overlapping pixels
        loss = torch.nn.functional.l1_loss(overlapping1, overlapping2)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track progress
        losses.append(loss.item())
        kernel_diff = torch.nn.functional.l1_loss(
            trainable_conv.weight.data, fixed_kernel
        ).item()
        kernel_diffs.append(kernel_diff)

        # Early stopping if converged
        if kernel_diff < 0.05:
            break

    # Final kernel difference
    final_kernel_diff = torch.nn.functional.l1_loss(
        trainable_conv.weight.data, fixed_kernel
    ).item()

    print(f"\nFinal kernel L1 difference: {final_kernel_diff:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Initial kernel diff: {kernel_diffs[0]:.6f}")
    print(f"Number of iterations: {len(losses)}")

    # Assertions - focus on kernel convergence
    assert (
        final_kernel_diff < 0.1
    ), f"Kernels did not converge. Final difference: {final_kernel_diff}"
    assert (
        kernel_diffs[-1] < kernel_diffs[0] * 0.5
    ), f"Kernel difference did not decrease significantly. Initial: {kernel_diffs[0]:.6f}, Final: {final_kernel_diff:.6f}"

    # Verify that trainable conv has gradients
    assert (
        trainable_conv.weight.grad is not None
    ), "Trainable conv weights have no gradients"
    assert (
        torch.abs(trainable_conv.weight.grad).sum().item() > 0
    ), "Trainable conv weight gradients are zero"


def test_alternating_kernel_optimization_3d():
    """
    Test alternating optimization between two 3D kernels.
    """
    torch.manual_seed(999)

    # Create input tensor: (3, 1, 32, 32, 32)
    input_tensor = torch.randn(3, 1, 32, 32, 32)

    # Extract patch pairs with 9x9x9 window
    window = (9, 9, 9)  # max_window=9, so delta_range must be in [2, 6]
    num_patches = 2  # 2 patch pairs per volume = 6 total pairs
    delta_range = (3.0, 5.0)

    patches1, patches2, deltas = extract_patch_pairs_3d(
        input_tensor, window, num_patches, delta_range, random_seed=999
    )

    # Detach patches for optimization
    patches1 = patches1.detach()
    patches2 = patches2.detach()

    # Verify that overlap fraction is less than 100%
    overlapping1_check, overlapping2_check = extract_overlapping_pixels_3d(
        patches1, patches2, deltas
    )
    total_pixels_per_patch = window[0] * window[1] * window[2]  # 9*9*9 = 729
    total_patches = patches1.shape[0]  # 6 patches
    total_possible_pixels = total_patches * total_pixels_per_patch  # 6 * 729 = 4374
    overlapping_pixels_count = overlapping1_check.shape[0]  # K

    overlap_fraction = overlapping_pixels_count / total_possible_pixels

    print("\nOverlap statistics:")
    print(f"  Total patches: {total_patches}")
    print(f"  Pixels per patch: {total_pixels_per_patch}")
    print(f"  Total possible pixels: {total_possible_pixels}")
    print(f"  Overlapping pixels: {overlapping_pixels_count}")
    print(f"  Overlap fraction: {overlap_fraction:.4f} ({overlap_fraction*100:.2f}%)")

    # Verify overlap is less than 100%
    assert (
        overlap_fraction < 1.0
    ), f"Overlap fraction should be less than 100%, got {overlap_fraction*100:.2f}%"

    # Create two trainable Conv3D layers
    kernel_size = 3
    padding = kernel_size // 2

    conv1 = nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode="reflect",
        bias=False,
    )

    conv2 = nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode="reflect",
        bias=False,
    )

    # Initialize kernels with different values
    with torch.no_grad():
        conv1.weight.data = torch.randn_like(conv1.weight.data) * 0.5
        conv2.weight.data = torch.randn_like(conv2.weight.data) * 0.8

    # Store initial kernel values
    initial_kernel1 = conv1.weight.data.clone()
    initial_kernel2 = conv2.weight.data.clone()
    initial_diff = torch.nn.functional.l1_loss(initial_kernel1, initial_kernel2).item()

    # Set up optimizers for both kernels
    optimizer1 = torch.optim.Adam(conv1.parameters(), lr=0.05)
    optimizer2 = torch.optim.Adam(conv2.parameters(), lr=0.05)

    # Alternating optimization
    num_rounds = 5  # 5 rounds of alternating
    iterations_per_round = 5

    losses = []
    kernel_diffs = []

    for _ in range(num_rounds):
        # Optimize kernel 1
        for _ in range(iterations_per_round):
            optimizer1.zero_grad()

            output1 = conv1(patches1)
            output2 = conv2(patches2)

            overlapping1, overlapping2 = extract_overlapping_pixels_3d(
                output1, output2, deltas
            )

            loss = torch.nn.functional.l1_loss(overlapping1, overlapping2)
            loss.backward()
            optimizer1.step()

            losses.append(loss.item())
            kernel_diff = torch.nn.functional.l1_loss(
                conv1.weight.data, conv2.weight.data
            ).item()
            kernel_diffs.append(kernel_diff)

        # Optimize kernel 2
        for _ in range(iterations_per_round):
            optimizer2.zero_grad()

            output1 = conv1(patches1)
            output2 = conv2(patches2)

            overlapping1, overlapping2 = extract_overlapping_pixels_3d(
                output1, output2, deltas
            )

            loss = torch.nn.functional.l1_loss(overlapping1, overlapping2)
            loss.backward()
            optimizer2.step()

            losses.append(loss.item())
            kernel_diff = torch.nn.functional.l1_loss(
                conv1.weight.data, conv2.weight.data
            ).item()
            kernel_diffs.append(kernel_diff)

    # Final kernel difference
    final_kernel_diff = torch.nn.functional.l1_loss(
        conv1.weight.data, conv2.weight.data
    ).item()

    print(f"\nInitial kernel L1 difference: {initial_diff:.6f}")
    print(f"Final kernel L1 difference: {final_kernel_diff:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Total optimization steps: {len(losses)}")

    # Verify that kernels converged toward each other
    assert (
        final_kernel_diff < initial_diff * 0.5
    ), f"Kernels did not converge. Initial diff: {initial_diff:.6f}, Final diff: {final_kernel_diff:.6f}"
