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
    'direct': [-1, 0, 1],  # Extract z-1, z, z+1 as separate channels
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
# 3 direct + 2 mean = 5 channels per input channel
```

## Key Features

- **Flexible Channel Specifications**: Extract specific z-slices or compute aggregations (mean, future: max, min, median)
- **Multiple Data Sources**: Works with torch.Tensor, zarr, HDF5, memory-mapped arrays
- **Two Accumulation Modes**: 
  - `"2d"`: Flatten to (N, C', Y, X) for 2D processing
  - `"3d"`: Keep as (N, C', Z, Y, X) to preserve 3D structure
- **Selective Slicing**: Process only specific z-slices
- **Parallelization Support**: Extraction and stitching plans with color groups
- **2D Integration**: Direct integration with 2D patch pair extraction

## Channel Specifications

### Direct Indexing

Extract specific z-slices as separate channels:

```python
spec = {'direct': [-2, -1, 0, 1, 2]}  # 5 channels
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

### Combined Operations

```python
spec = {
    'direct': [-1, 0, 1],
    'mean': [[-2, -3], [2, 3]]
}
# Total: 3 + 2 = 5 channels per input channel
```

## Data Sources

### In-Memory Tensor

```python
data = torch.randn(10, 3, 50, 200, 200)
quilt = NCZYX25DQuilt(data, channel_spec={'direct': [0]})
```

### Zarr Files

```python
# Using convenience function
from qlty.backends_2_5D import from_zarr

data = from_zarr("data.zarr")
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'direct': [-1, 0, 1]}
)

# Or manually
import zarr
from qlty.backends_2_5D import ZarrBackend, TensorLike3D

z = zarr.open("data.zarr", mode='r')
backend = ZarrBackend(z)
data = TensorLike3D(backend)
quilt = NCZYX25DQuilt(data, channel_spec={'direct': [0]})
```

### HDF5 Files

```python
from qlty.backends_2_5D import from_hdf5

data = from_hdf5("data.h5", dataset_path="/images/stack")
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'direct': [-1, 0, 1]}
)
```

## Extract Patch Pairs

Integrate with 2D patch pair extraction:

```python
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'direct': [-1, 0, 1]},
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
    channel_spec={'direct': [-5, 0, 5]},
    boundary_mode="reflect"  # Mirror at boundaries
)
```

## API Reference

### NCZYX25DQuilt

Main class for 2.5D conversion.

**Methods:**
- `convert()`: Convert 3D data to 2.5D
- `extract_patch_pairs()`: Extract patch pairs using 2D interface
- `create_extraction_plan()`: Create extraction plan for parallelization
- `create_stitching_plan()`: Create stitching plan
- `get_channel_metadata()`: Get metadata for each output channel
- `validate_spec()`: Validate channel specification
- `to_ncyx_quilt()`: Convert to 2D quilt

### ZOperation

Enum for channel operations:
- `ZOperation.DIRECT`: Single pixel extraction
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

