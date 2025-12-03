# Plan: Laplacian Pyramid Difference Maps for OME-Zarr

## Overview
Add optional Laplacian pyramid mode to `stack_files_to_ome_zarr` that stores difference maps (residuals) at each resolution level instead of (or alongside) the downsampled images. This enables perfect reconstruction from the lowest resolution plus all difference maps.

## Current Implementation Analysis

### Current Flow (`_load_and_write_to_all_pyramid_levels`)
1. Load image slice (Y, X) or (C, Y, X)
2. Write to base level (level 0) - full resolution
3. For each pyramid level (1, 2, 3, ...):
   - Downsample current image using block averaging
   - Write downsampled image to that level
   - Use downsampled image as input for next level

### Current Downsampling Method
- Uses block averaging (numpy reshape + mean)
- Handles padding for non-divisible dimensions
- Supports both single-channel (Y, X) and multi-channel (C, Y, X)

**Note**: Will be converted to PyTorch `avg_pool2d` for consistency with upsampling.

## Proposed Laplacian Pyramid Implementation

### Concept
A Laplacian pyramid stores:
- **Level 0**: Difference map for highest resolution (full resolution)
- **Level 1-N-1**: Difference maps for progressively lower resolutions
- **Level N (highest level number)**: The most downsampled version (coarsest resolution, base level)

Following standard pyramid convention: **Level 0 = highest resolution**, higher level numbers = lower resolution.

This allows perfect reconstruction:
```
reconstructed = upsample(base_level) + sum(all_difference_maps)
```

Note: The base level (lowest resolution Gaussian level) is stored at the highest level number to match the standard convention where level 0 represents the highest resolution.

### Architecture Options

#### Option A: New Function (Recommended)
Create `stack_files_to_ome_zarr_laplacian()` as a fresh implementation:
- **Pros**: 
  - Clean separation of concerns
  - Doesn't risk breaking existing functionality
  - Easier to test and maintain
  - Can be optimized independently
- **Cons**: 
  - Some code duplication
  - Need to maintain two code paths

#### Option B: Add Mode Flag
Add `pyramid_mode: str = "gaussian" | "laplacian"` parameter to existing function:
- **Pros**: 
  - Single entry point
  - Shared infrastructure
- **Cons**: 
  - More complex conditional logic
  - Risk of breaking existing code
  - Harder to test edge cases

**Recommendation: Option A** - Fresh implementation for safety and clarity.

## Implementation Plan

### 1. New Function: `stack_files_to_ome_zarr_laplacian()`

**Location**: `qlty/utils/stack_to_zarr.py`

**Signature**:
```python
def stack_files_to_ome_zarr_laplacian(
    directory: str | Path,
    extension: str,
    pattern: str | re.Pattern,
    output_dir: str | Path | None = None,
    zarr_chunks: tuple[int, ...] | None = None,
    dtype: np.dtype | None = None,
    axis_order: str = "ZCYX",
    output_naming: Callable[[str], str] | None = None,
    sort_by_counter: bool = True,
    dry_run: bool = False,
    num_workers: int | None = None,
    pyramid_levels: int | None = None,
    pyramid_scale_factors: list[tuple[int, ...]] | None = None,
    downsample_mode: str = "2d",
    downsample_axes: tuple[str, ...] | None = None,
    interpolation_mode: str = "bilinear",  # "bilinear" or "bicubic"
    store_base_level: bool = True,  # Whether to store lowest resolution level
    verbose: bool = True,
) -> dict[str, dict]:
```

**Key Parameters**:
- `interpolation_mode`: "bilinear" or "bicubic" for upsampling
- `store_base_level`: If True, store the lowest resolution level; if False, only store differences

### 2. Core Worker Function: `_load_and_write_laplacian_pyramid()`

**Process**:
1. Load image slice
2. Build Gaussian pyramid (downsample progressively using PyTorch `avg_pool2d`)
3. Build Laplacian pyramid (compute differences):
   - Start from lowest resolution level
   - For each level going up:
     - Upsample lower level using torch interpolation
     - Compute difference: `current_level - upsampled_lower_level`
     - Store difference map
4. Write all levels to Zarr

**Downsampling with PyTorch**:
```python
import torch
import torch.nn.functional as F

# Convert numpy to torch tensor
img_torch = torch.from_numpy(img_numpy).float()

# Downsample using average pooling (block averaging)
# For scale factor of 2x: kernel_size=2, stride=2
# Input: (C, Y, X) -> Output: (C, Y//2, X//2)
downsampled = F.avg_pool2d(
    img_torch.unsqueeze(0),  # Add batch dim: (1, C, Y, X)
    kernel_size=(y_scale_int, x_scale_int),
    stride=(y_scale_int, x_scale_int),
    padding=0,  # Padding handled separately if needed
)
downsampled = downsampled.squeeze(0).numpy()  # Remove batch dim, back to numpy
```

**Upsampling with PyTorch**:
```python
import torch
import torch.nn.functional as F

# Convert numpy to torch tensor
img_torch = torch.from_numpy(img_numpy).float()

# Upsample: (C, Y, X) -> (C, Y_new, X_new)
upsampled = F.interpolate(
    img_torch.unsqueeze(0),  # Add batch dim: (1, C, Y, X)
    size=(Y_target, X_target),
    mode=interpolation_mode,  # "bilinear" or "bicubic"
    align_corners=False,  # For bilinear
    antialias=True,  # For bicubic (if available)
)
upsampled = upsampled.squeeze(0).numpy()  # Remove batch dim, back to numpy
```

**Difference Computation**:
```python
# For level i (going from lowest to highest):
current_level = gaussian_pyramid[i]  # Already downsampled
lower_level = gaussian_pyramid[i-1]  # One level lower

# Upsample lower level to match current level's shape
upsampled_lower = upsample(lower_level, size=current_level.shape[-2:])

# Compute difference
difference = current_level - upsampled_lower

# Store difference (or store both if store_base_level=True)
```

### 3. Zarr Storage Structure

**Option 1: Store differences in separate arrays**
```
ome.zarr/
├── diff_0/     # Difference map level 0 (highest resolution)
├── diff_1/     # Difference map level 1
├── diff_2/     # Difference map level 2
└── N/          # Base level (lowest resolution, highest level number)
```

Following standard convention: Level 0 = highest resolution, Level N = lowest resolution.

**Option 2: Store differences alongside Gaussian pyramid**
```
ome.zarr/
├── 0/          # Full resolution (Gaussian)
├── 1/          # Downsampled (Gaussian)
├── 2/          # More downsampled (Gaussian)
├── laplacian_1/  # Difference map level 1
└── laplacian_2/  # Difference map level 2
```

**Recommendation: Option 1** - Cleaner separation, easier reconstruction.

### 4. Reconstruction Function

Add utility function to reconstruct full resolution from Laplacian pyramid:
```python
def reconstruct_from_laplacian_pyramid(
    zarr_group_path: str | Path,
    z_idx: int | None = None,  # If None, reconstruct all slices
) -> np.ndarray:
    """
    Reconstruct full resolution image from Laplacian pyramid.
    
    Process:
    1. Load base level (lowest resolution)
    2. Upsample and add difference maps progressively
    3. Return reconstructed full resolution
    """
```

### 5. Integration Points

**Shared Infrastructure** (reuse from existing code):
- `_load_and_process_image()` - Image loading
- `_apply_axis_order()` - Axis reordering
- `_create_zarr_array()` - Zarr array creation
- File discovery and grouping logic
- Multiprocessing infrastructure

**New Components**:
- `_load_and_write_laplacian_pyramid()` - Main worker function
- `_downsample_with_torch()` - Downsampling helper using avg_pool2d
- `_upsample_with_torch()` - Upsampling helper using interpolate
- `_compute_laplacian_levels()` - Difference computation
- `reconstruct_from_laplacian_pyramid()` - Reconstruction utility

### 6. Implementation Steps

1. **Phase 1: Core Functionality**
   - Implement `_downsample_with_torch()` helper using `avg_pool2d`
   - Implement `_upsample_with_torch()` helper using `interpolate`
   - Implement `_load_and_write_laplacian_pyramid()` worker
   - Test on single slice with 2-3 pyramid levels

2. **Phase 2: Integration**
   - Implement `stack_files_to_ome_zarr_laplacian()` main function
   - Integrate with existing file discovery and multiprocessing
   - Handle axis orders and multi-channel images

3. **Phase 3: Reconstruction**
   - Implement `reconstruct_from_laplacian_pyramid()` utility
   - Add validation: `reconstructed == original` (within numerical precision)

4. **Phase 4: Testing & Documentation**
   - Unit tests for upsampling and difference computation
   - Integration tests for full pipeline
   - Performance benchmarks vs. Gaussian pyramid
   - Documentation with examples

### 7. Technical Considerations

**Memory**:
- Laplacian pyramid requires storing all levels temporarily
- For large images, may need to process in chunks
- Consider memory-mapped arrays for intermediate storage

**Precision**:
- Difference maps may have negative values
- Consider using `int16` or `float32` instead of `uint8`
- Or store as signed integers with offset

**Downsampling**:
- Use `torch.nn.functional.avg_pool2d` for block averaging
- Equivalent to numpy reshape + mean but more efficient
- Handles padding consistently with upsampling

**Interpolation**:
- Bilinear: Faster, good for most cases
- Bicubic: Better quality, slower
- `align_corners=False` recommended for better edge handling

**Axis Handling**:
- Support same axis orders as current implementation
- Handle Z, C, Y, X dimensions correctly
- Only upsample Y, X dimensions (not Z or C)

**Padding**:
- Need consistent padding strategy between downsampling and upsampling
- May need to crop after reconstruction to match original size

### 8. Example Usage

```python
from qlty.utils.stack_to_zarr import (
    stack_files_to_ome_zarr_laplacian,
    reconstruct_from_laplacian_pyramid,
)

# Create Laplacian pyramid OME-Zarr
result = stack_files_to_ome_zarr_laplacian(
    directory="/path/to/images",
    extension=".tif",
    pattern=r"(.+)_(\d+)\.tif$",
    pyramid_levels=4,
    interpolation_mode="bilinear",
    store_base_level=True,
)

# Reconstruct full resolution from Laplacian pyramid
zarr_path = result["stack_name"]["zarr_path"]
reconstructed = reconstruct_from_laplacian_pyramid(zarr_path, z_idx=0)
```

### 9. Testing Strategy

**Unit Tests**:
- `_downsample_with_torch()`: Test avg_pool2d downsampling vs numpy reshape+mean
- `_upsample_with_torch()`: Test bilinear/bicubic upsampling
- `_compute_laplacian_levels()`: Test difference computation
- Verify perfect reconstruction: `original == reconstructed`
- Verify PyTorch downsampling matches numpy implementation

**Integration Tests**:
- Full pipeline with single-channel images
- Full pipeline with multi-channel images
- Different axis orders
- Different pyramid levels
- Edge cases: small images, large images, non-square

**Performance Tests**:
- Compare storage size: Gaussian vs. Laplacian
- Compare reconstruction time
- Memory usage profiling

### 10. Open Questions

1. **Storage Format**: Should differences be stored as separate arrays or alongside Gaussian pyramid?
2. **Data Type**: What dtype for difference maps? (may need signed types)
3. **Compression**: Should difference maps use different compression?
4. **Metadata**: How to indicate Laplacian pyramid mode in OME-Zarr metadata?
5. **Backward Compatibility**: Should we add a flag to existing function instead?

## Recommendation

**Proceed with Option A (new function)** for the following reasons:
1. Safety: Doesn't risk breaking existing functionality
2. Clarity: Clean separation makes code easier to understand
3. Testing: Easier to test independently
4. Performance: Can optimize specifically for Laplacian pyramid workflow

**Next Steps**:
1. Get approval on this plan
2. Implement Phase 1 (core functionality)
3. Test and iterate
4. Integrate with main pipeline
5. Add reconstruction utility
6. Comprehensive testing and documentation

