# 2.5D Quilt Module

The `qlty2_5D` module provides functionality to convert 3D volumetric data into 2.5D multi-channel data by slicing the Z dimension into channels. This enables processing 3D data with 2D methods while preserving spatial context from neighboring slices.

## Quick Start

```python
import torch
from qlty.qlty2_5D import NCZYX25DQuilt

# Create 3D data: (N, C, Z, Y, X)
data = torch.randn(5, 1, 20, 100, 100)

# Define channel specification
spec = {
    'identity': [-1, 0, 1],  # Extract z-1, z, z+1 as separate channels
    'mean': [[-2, -3], [2, 3]]  # Mean of slices below and above
}

# Create quilt and convert
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec=spec,
    accumulation_mode="2d",  # Flatten to 2D
    z_slices=[5, 6, 7, 8, 9]  # Process only these z-slices
)

# Convert to 2.5D
result = quilt.convert()  # Shape: (5, 5, 100, 100)
# 5 z-slices * 1 image = 5 separate 2D images
# 3 identity + 2 mean = 5 channels per input channel
```

## Key Features

- **Flexible Channel Specifications**: Extract specific z-slices or compute aggregations (mean, future: max, min, median)
- **Multiple Data Sources**: Works with torch.Tensor, zarr, HDF5, memory-mapped arrays
- **Two Accumulation Modes**:
  - `"2d"`: Each z-slice becomes a separate 2D image: (N*Z_selected, C', Y, X)
  - `"3d"`: Keep Z dimension separate: (N, C', Z_selected, Y, X)
- **Selective Slicing**: Process only specific z-slices
- **Parallelization Support**: Extraction and stitching plans with color groups
- **2D Integration**: Direct integration with 2D patch pair extraction

## Channel Specifications

### Identity Indexing

Extract specific z-slices as separate channels:

```python
spec = {'identity': [-2, -1, 0, 1, 2]}  # 5 channels
```

### Mean Aggregation

Compute mean of multiple z-slices:

```python
spec = {
    'mean': [
        [-1, -2, -3],  # Channel 1: mean(z-1, z-2, z-3)
        [1, 2, 3]       # Channel 2: mean(z+1, z+2, z+3)
    ]
}
```

### Standard Deviation

Compute standard deviation of multiple z-slices:

```python
spec = {
    'std': [
        [-1, -2, -3],  # Channel 1: std(z-1, z-2, z-3)
        [1, 2, 3]       # Channel 2: std(z+1, z+2, z+3)
    ]
}
```

### Combined Operations

```python
spec = {
    'identity': [-1, 0, 1],
    'mean': [[-2, -3], [2, 3]],
    'std': [[-1, 0, 1]]
}
# Total: 3 + 2 + 1 = 6 channels per input channel
```

## Data Sources

### In-Memory Tensor

```python
data = torch.randn(10, 3, 50, 200, 200)
quilt = NCZYX25DQuilt(data, channel_spec={'identity': [0]})
```

### Zarr Files

```python
# Using convenience function
from qlty.backends_2_5D import from_zarr

data = from_zarr("data.zarr")
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'identity': [-1, 0, 1]}
)

# Or manually
import zarr
from qlty.backends_2_5D import ZarrBackend, TensorLike3D

z = zarr.open("data.zarr", mode='r')
backend = ZarrBackend(z)
data = TensorLike3D(backend)
quilt = NCZYX25DQuilt(data, channel_spec={'identity': [0]})
```

### HDF5 Files

```python
from qlty.backends_2_5D import from_hdf5

data = from_hdf5("data.h5", dataset_path="/images/stack")
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'identity': [-1, 0, 1]}
)
```

## Extract Patch Pairs

Integrate with 2D patch pair extraction:

```python
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'identity': [-1, 0, 1]},
    accumulation_mode="2d"
)

# Extract patch pairs using 2D interface
patches1, patches2, deltas, rotations = quilt.extract_patch_pairs(
    window=(32, 32),
    num_patches=100,
    delta_range=(8.0, 16.0),
    random_seed=42
)
```

## Extract Overlapping Pixels

Extract overlapping pixels from patch pairs:

```python
# Direct method - combines patch extraction and overlap extraction
overlapping1, overlapping2 = quilt.extract_overlapping_pixels(
    window=(32, 32),
    num_patches=100,
    delta_range=(8.0, 16.0),
    random_seed=42
)

# overlapping1[i] and overlapping2[i] correspond to the same spatial location
```

## Parallelization

Create extraction and stitching plans for parallel processing:

```python
# Create plans
extraction_plan = quilt.create_extraction_plan(
    window=(64, 64),
    step=(32, 32),
    color_y_mod=4,
    color_x_mod=4
)

stitching_plan = quilt.create_stitching_plan(extraction_plan)

# Process by color group (can be parallelized)
for color_y in range(4):
    for color_x in range(4):
        patches = extraction_plan.get_patches_for_color(color_y, color_x)
        # Process patches independently (no race conditions)
        results = process_patches(patches)
        # Stitch results
        stitch_patches(stitching_plan, color_y, color_x, results)
```

## Just-In-Time (JIT) Extraction

The extraction plan system supports **just-in-time (JIT) loading** - it doesn't cache the entire converted dataset. Instead, it loads only the required z-slices and spatial regions when you extract each patch:

```python
# Create extraction plan (no data loaded yet - just metadata)
extraction_plan = quilt.create_extraction_plan(
    window=(64, 64),
    step=(32, 32)
)

# Extract patches on-demand (JIT loading)
for patch_spec in extraction_plan.patches[:10]:  # Only first 10 patches
    # This loads ONLY the required z-slices for this patch
    # No full dataset conversion or caching
    patch_data = quilt.extract_patch_from_plan(patch_spec)
    # patch_data shape: (C', 64, 64)
    # Process patch...
```

**Key Benefits:**
- **Memory efficient**: Only loads what you need, when you need it
- **No upfront conversion**: Doesn't convert entire 3D→2.5D dataset upfront
- **Backend-aware**: Works with zarr/HDF5 backends that load on-demand
- **Selective processing**: Process only a subset of patches without loading everything

**How it works:**
1. `create_extraction_plan()` creates metadata (no data loading)
2. Each patch specifies `required_z_indices` - exactly which z-slices are needed
3. `extract_patch_from_plan()` loads only those z-slices on-demand
4. Channel operations are applied to the loaded chunk
5. Result is cropped to patch window if specified

## Non-Overlapping Colored Patches (Batching)

For batching scenarios where you want non-overlapping patches within each batch, use `create_non_overlapping_colored_plan()`:

```python
# Create plan with non-overlapping patches per color group
plan = quilt.create_non_overlapping_colored_plan(
    window=(64, 64),
    step=(32, 32)  # 50% overlap overall, but non-overlapping within groups
)

# Process each color group as a batch (no overlap within batch)
for color_y in range(max(p.color_y_idx for p in plan.patches) + 1):
    for color_x in range(max(p.color_x_idx for p in plan.patches) + 1):
        patches = plan.get_patches_for_color(color_y, color_x)
        # All patches in this batch are non-overlapping
        batch = torch.stack([quilt.extract_patch_from_plan(p) for p in patches])
        # batch shape: (num_patches, C', 64, 64)
        # Process batch (no overlap concerns)...
```

**How it works:**
- Color groups are computed such that `color_y_mod = ceil(window[0] / step[0])`
- This ensures patches in the same color group are spaced by at least `window[0]` pixels apart
- Patches within a color group are guaranteed non-overlapping, perfect for batching
- Different color groups can overlap with each other (for full coverage)

## Integration with 2D Quilt

Convert to 2D quilt for further processing:

```python
# Convert and create 2D quilt
quilt_2d = quilt.to_ncyx_quilt(
    window=(64, 64),
    step=(32, 32),
    border=(8, 8)
)

# Use 2D quilt operations
converted = quilt.convert()  # (N, C', Y, X)
patches = quilt_2d.unstitch(converted)
# Process patches...
reconstructed = quilt_2d.stitch(processed_patches)
```

## Boundary Handling

Handle out-of-bounds z-slices with different modes:

- `"clamp"` (default): Repeat edge slices
- `"zero"`: Zero-padding
- `"reflect"`: Mirror padding
- `"skip"`: Skip invalid slices

```python
quilt = NCZYX25DQuilt(
    data,
    channel_spec={'identity': [-5, 0, 5]},
    boundary_mode="reflect"  # Mirror at boundaries
)
```

## API Reference

### NCZYX25DQuilt

Main class for 2.5D conversion.

**Methods:**
- `convert()`: Convert 3D data to 2.5D (loads entire dataset)
- `extract_patch_pairs()`: Extract patch pairs using 2D interface
- `extract_patch_from_plan()`: Extract single patch using JIT loading (memory efficient)
- `create_extraction_plan()`: Create extraction plan for parallelization
- `create_non_overlapping_colored_plan()`: Create plan with non-overlapping patches per color group (for batching)
- `create_stitching_plan()`: Create stitching plan
- `get_channel_metadata()`: Get metadata for each output channel
- `validate_spec()`: Validate channel specification
- `to_ncyx_quilt()`: Convert to 2D quilt

### ZOperation

Enum for channel operations:
- `ZOperation.IDENTITY`: Single pixel extraction
- `ZOperation.MEAN`: Mean of multiple pixels

### Backends

- `InMemoryBackend`: Wraps torch.Tensor
- `MemoryMappedBackend`: Memory-mapped numpy arrays
- `ZarrBackend`: OME-Zarr files
- `HDF5Backend`: HDF5 files
- `TensorLike3D`: Wrapper that makes any backend look like a tensor

## Convenience Functions

The `qlty.backends_2_5D` module provides convenience functions for creating TensorLike3D:

- `from_zarr(zarr_path)`: Create from zarr file
- `from_hdf5(hdf5_path, dataset_path)`: Create from HDF5 file
- `from_memmap(file_path, dtype, shape)`: Create from memory-mapped array

## See Also

- `qlty.qlty2D.NCYXQuilt`: 2D quilt for processing converted data
- `qlty.patch_pairs_2d.extract_patch_pairs`: Patch pair extraction
- `qlty.backends_2_5D`: Backend implementations and convenience functions

## Examples

See `docs/qlty2_5D_examples.md` for comprehensive examples.
