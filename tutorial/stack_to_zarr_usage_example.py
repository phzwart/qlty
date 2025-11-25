"""
Example: Using stack_to_zarr output with 2.5D Quilt and patch pair extraction

This script demonstrates how to:
1. Load a zarr file created by stack_files_to_zarr()
2. Create 2.5D slices using NCZYX25DQuilt
3. Extract partially overlapping patch pairs from the 2.5D slices
"""

import torch
import zarr

from qlty.backends_2_5D import TensorLike3D, ZarrBackend
from qlty.patch_pairs_2d import extract_patch_pairs
from qlty.qlty2_5D import NCZYX25DQuilt

# ============================================================================
# Step 1: Load the zarr file created by stack_files_to_zarr()
# ============================================================================

# Path to your zarr file (created by stack_files_to_zarr)
zarr_path = (
    "./monet/monet_61389_16_A_pos18_bottom_newimgsys_prop405_exp0p05_0324_rec.zarr"
)

# Open the zarr array
zarr_array = zarr.open(zarr_path, mode="r")

# Print info about the zarr array

# ============================================================================
# Step 2: Wrap zarr array with TensorLike3D for use with 2.5D Quilt
# ============================================================================

# Create backend wrapper
# Note: stack_to_zarr creates arrays with shape (Z, Y, X) for single channel
# or (C, Z, Y, X) or (Z, C, Y, X) depending on axis_order
# We need to ensure it's in (N, C, Z, Y, X) format

# Check the shape and adjust if needed
shape = zarr_array.shape

# If single channel (Z, Y, X), reshape to (1, 1, Z, Y, X)
if len(shape) == 3:
    # Reshape view: treat as (1, 1, Z, Y, X)
    # Create backend
    backend = ZarrBackend(zarr_array, dtype=torch.float32)
    # The backend will automatically add N=1, C=1 dimensions

# If multi-channel with C first (C, Z, Y, X), reshape to (1, C, Z, Y, X)
elif (len(shape) == 4 and shape[0] <= 10) or len(shape) == 5:  # Likely (C, Z, Y, X)
    backend = ZarrBackend(zarr_array, dtype=torch.float32)

# Wrap backend in TensorLike3D interface
tensor_like = TensorLike3D(backend)

# Verify we can load data correctly
# Test load: should return (C, Z, Y, X) = (1, 800, 3232, 3232)
test_load = tensor_like[0]  # Get first image: (C, Z, Y, X)

# ============================================================================
# Step 3: Create 2.5D slices using NCZYX25DQuilt
# ============================================================================

# Define channel specification for 2.5D conversion
# Example 1: Extract z-1, z, z+1 as separate channels
channel_spec = {
    "identity": [-1, 0, 1],  # 3 channels: previous slice, current slice, next slice
}

# Example 2: More complex specification with mean operations
# channel_spec = {
#     'identity': [-1, 0, 1],  # 3 channels: direct slices
#     'mean': [
#         [-2, -3],  # Channel 4: mean of z-2 and z-3
#         [2, 3]      # Channel 5: mean of z+2 and z+3
#     ]
# }
# This creates 5 channels total

# Create 2.5D Quilt
# accumulation_mode="2d" flattens to (N*Z_selected, C', Y, X) - separate 2D images
# accumulation_mode="3d" keeps as (N, C', Z_selected, Y, X) - 3D structure
quilt = NCZYX25DQuilt(
    data_source=tensor_like,
    channel_spec=channel_spec,
    accumulation_mode="2d",  # Flatten to 2D images
    z_slices=slice(10, 790),  # Process z-slices 10 to 789 (adjust as needed)
    boundary_mode="clamp",  # Handle boundaries: "clamp", "zero", "reflect", "skip"
)

# Convert to 2.5D
result_2_5d = quilt.convert()

# ============================================================================
# Step 4: Extract partially overlapping patch pairs from 2.5D slices
# ============================================================================

# Extract patch pairs with controlled displacement
# Each pair consists of two patches with a specific displacement range

window = (64, 64)  # Patch size: 64x64 pixels
num_patches = 100  # Number of patch pairs per image
delta_range = (16.0, 32.0)  # Euclidean distance between patch centers
# delta_range means: 16 <= sqrt(dx² + dy²) <= 32 pixels


patches1, patches2, deltas, rotations = extract_patch_pairs(
    tensor=result_2_5d,
    window=window,
    num_patches=num_patches,
    delta_range=delta_range,
    random_seed=42,  # For reproducibility
    rotation_choices=None,  # Or [0, 1, 2, 3] for quarter-turn rotations
)


# ============================================================================
# Step 5: Use the patches for training or analysis
# ============================================================================

# Example: Visualize first patch pair

# Get first patch pair
patch1 = patches1[0]  # Shape: (C', 64, 64)
patch2 = patches2[0]  # Shape: (C', 64, 64)
delta = deltas[0]  # Shape: (2,) - displacement vector


# Visualize (if you have matplotlib and the patches are reasonable size)
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(patch1[0].numpy(), cmap='gray')  # Show first channel
# axes[0].set_title('Patch 1')
# axes[1].imshow(patch2[0].numpy(), cmap='gray')  # Show first channel
# axes[1].set_title(f'Patch 2 (dx={delta[0]:.1f}, dy={delta[1]:.1f})')
# plt.tight_layout()
# plt.show()
