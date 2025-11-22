# 2.5D Quilt Examples

This document provides examples for using the 2.5D Quilt implementation to convert 3D data into 2.5D multi-channel data.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Channel Specifications](#channel-specifications)
3. [Working with Different Data Sources](#working-with-different-data-sources)
4. [Extracting Patch Pairs](#extracting-patch-pairs)
5. [Parallelization with Plans](#parallelization-with-plans)
6. [Integration with 2D Quilt](#integration-with-2d-quilt)

## Basic Usage

### Simple 2.5D Conversion

```python
import torch
from qlty.qlty2_5D import NCZYX25DQuilt

# Create 3D data: (N, C, Z, Y, X)
data = torch.randn(5, 1, 20, 100, 100)  # 5 images, 1 channel, 20 z-slices

# Define channel specification
# Extract z-1, z, z+1 as separate channels
spec = {
    'direct': [-1, 0, 1]
}

# Create quilt and convert
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec=spec,
    accumulation_mode="2d",  # Flatten to 2D
    z_slices=[5, 6, 7, 8, 9]  # Process only these z-slices
)

# Convert to 2.5D
result = quilt.convert()  # Shape: (5, 3, 100, 100)
# 3 channels: z-1, z, z+1 for each processed z-slice
```

### 3D Accumulation Mode

```python
# Keep Z dimension separate
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'direct': [-1, 0, 1]},
    accumulation_mode="3d",  # Keep 3D structure
    z_slices=[5, 6, 7]
)

result = quilt.convert()  # Shape: (5, 3, 3, 100, 100)
# 3 channels × 3 z-slices = 9 total channels per image
```

## Channel Specifications

### Direct Indexing

Extract specific z-slices as separate channels:

```python
spec = {
    'direct': [-2, -1, 0, 1, 2]  # 5 channels: z-2, z-1, z, z+1, z+2
}
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

Mix direct and mean operations:

```python
spec = {
    'direct': [-1, 0, 1],           # 3 channels: z-1, z, z+1
    'mean': [
        [-2, -3, -4],               # 1 channel: mean(z-2, z-3, z-4)
        [2, 3, 4]                    # 1 channel: mean(z+2, z+3, z+4)
    ]
}
# Total: 5 channels per input channel
```

### Using Enum Format

```python
from qlty.qlty2_5D import ZOperation

spec = {
    ZOperation.DIRECT: (-1, 0, 1),
    ZOperation.MEAN: ((-2, -3), (2, 3))
}
```

## Working with Different Data Sources

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
    channel_spec={'direct': [-1, 0, 1]},
    accumulation_mode="2d"
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
# Using convenience function
from qlty.backends_2_5D import from_hdf5

data = from_hdf5("data.h5", dataset_path="/images/stack")
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'direct': [-1, 0, 1]},
    accumulation_mode="2d"
)
```

### Memory-Mapped Arrays

```python
import numpy as np
from qlty.backends_2_5D import MemoryMappedBackend, TensorLike3D

mmap = np.memmap("data.dat", dtype='float32', mode='r', shape=(N, C, Z, Y, X))
backend = MemoryMappedBackend(mmap)
data = TensorLike3D(backend)
quilt = NCZYX25DQuilt(data, channel_spec={'direct': [0]})
```

## Extracting Patch Pairs

The 2.5D quilt integrates with the 2D patch pair extraction:

```python
# Convert 3D to 2.5D and extract patch pairs in one step
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'direct': [-1, 0, 1]},
    accumulation_mode="2d"
)

# Extract patch pairs using 2D interface
patches1, patches2, deltas, rotations = quilt.extract_patch_pairs(
    window=(32, 32),        # Patch size
    num_patches=100,        # 100 pairs per image
    delta_range=(8.0, 16.0),  # Displacement distance
    random_seed=42
)

# patches1: (N*100, C', 32, 32)
# patches2: (N*100, C', 32, 32)
# deltas: (N*100, 2) - displacement vectors
# rotations: (N*100,) - quarter-turn rotations
```

## Extracting Overlapping Pixels

Extract overlapping pixels from patch pairs in one step:

```python
# Extract overlapping pixels directly
overlapping1, overlapping2 = quilt.extract_overlapping_pixels(
    window=(32, 32),
    num_patches=100,
    delta_range=(8.0, 16.0),
    random_seed=42
)

# overlapping1: (K, C') - overlapping pixels from first patches
# overlapping2: (K, C') - overlapping pixels from second patches
# K is the total number of overlapping pixels across all pairs
# overlapping1[i] and overlapping2[i] correspond to the same spatial location
```

Or extract patch pairs first, then get overlapping pixels:

```python
# Step 1: Extract patch pairs
patches1, patches2, deltas, rotations = quilt.extract_patch_pairs(
    window=(32, 32),
    num_patches=100,
    delta_range=(8.0, 16.0)
)

# Step 2: Extract overlapping pixels
from qlty.patch_pairs_2d import extract_overlapping_pixels
overlapping1, overlapping2 = extract_overlapping_pixels(
    patches1, patches2, deltas, rotations
)
```

## Parallelization with Plans

### Creating Extraction and Stitching Plans

```python
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'direct': [-1, 0, 1]},
    accumulation_mode="2d"
)

# Create extraction plan with color groups for parallelization
extraction_plan = quilt.create_extraction_plan(
    window=(64, 64),
    step=(32, 32),
    color_y_mod=4,  # 4 color groups in Y
    color_x_mod=4   # 4 color groups in X
)

# Create stitching plan
stitching_plan = quilt.create_stitching_plan(extraction_plan)

# Serialize plans for distributed processing
extraction_dict = extraction_plan.serialize()
stitching_dict = stitching_plan.serialize()
```

### Processing by Color Group

```python
# Process each color group independently (can be parallelized)
for color_y in range(4):
    for color_x in range(4):
        # Get patches for this color group
        patches = extraction_plan.get_patches_for_color(color_y, color_x)
        
        # Process patches (no race conditions - color groups don't overlap)
        results = []
        for patch in patches:
            # Load data for this patch
            z_data = data[patch.n, :, patch.required_z_indices, :, :]
            # Apply channel operations...
            result = process_patch(z_data, patch)
            results.append(result)
        
        # Stitch results for this color group
        stitch_patches(stitching_plan, color_y, color_x, results)
```

## Integration with 2D Quilt

### Converting to 2D Quilt

```python
# Convert 3D to 2.5D and create 2D quilt
quilt_2_5d = NCZYX25DQuilt(
    data_source=data,
    channel_spec={'direct': [-1, 0, 1]},
    accumulation_mode="2d"
)

# Get converted data and create 2D quilt
converted = quilt_2_5d.convert()  # (N, C', Y, X)

from qlty.qlty2D import NCYXQuilt
quilt_2d = NCYXQuilt(
    Y=converted.shape[2],
    X=converted.shape[3],
    window=(64, 64),
    step=(32, 32),
    border=(8, 8)
)

# Use 2D quilt on converted data
patches = quilt_2d.unstitch(converted)
# Process patches...
reconstructed = quilt_2d.stitch(processed_patches)
```

### Using Convenience Method

```python
quilt_2_5d = NCZYX25DQuilt(data, channel_spec={'direct': [0]}, accumulation_mode="2d")
quilt_2d = quilt_2_5d.to_ncyx_quilt(
    window=(64, 64),
    step=(32, 32),
    border=(8, 8)
)
```

## Boundary Handling

Different boundary modes handle out-of-bounds z-slices:

```python
# Clamp (default): Repeat edge slices
quilt = NCZYX25DQuilt(
    data, channel_spec={'direct': [-5, 0, 5]},
    boundary_mode="clamp"  # z-5 -> z=0, z+5 -> z=Z-1
)

# Zero: Zero-padding
quilt = NCZYX25DQuilt(
    data, channel_spec={'direct': [-5, 0, 5]},
    boundary_mode="zero"  # Out-of-bounds slices become zeros
)

# Reflect: Mirror padding
quilt = NCZYX25DQuilt(
    data, channel_spec={'direct': [-5, 0, 5]},
    boundary_mode="reflect"  # Mirror at boundaries
)

# Skip: Skip invalid slices
quilt = NCZYX25DQuilt(
    data, channel_spec={'direct': [-5, 0, 5]},
    boundary_mode="skip"  # Reduce channel count at boundaries
)
```

## Real-World Example: Cryo-EM Data

```python
from qlty.backends_2_5D import from_zarr
from qlty.qlty2_5D import NCZYX25DQuilt

# Load cryo-EM tomogram from zarr
data = from_zarr("tomogram.zarr")
quilt = NCZYX25DQuilt(
    data_source=data,
    channel_spec={
        'direct': [-2, -1, 0, 1, 2],  # 5 slices around center
        'mean': [
            [-3, -4, -5],  # Mean of 3 slices below
            [3, 4, 5]       # Mean of 3 slices above
        ]
    },
    accumulation_mode="2d",
    z_slices=range(10, 90)  # Process middle 80 slices
)

# Convert to 2.5D
converted = quilt.convert()  # (1, 7, Y, X) - 7 channels total

# Extract patch pairs for training
patches1, patches2, deltas, rotations = quilt.extract_patch_pairs(
    window=(128, 128),
    num_patches=1000,
    delta_range=(16.0, 32.0),
    random_seed=42
)
```

## Performance Tips

1. **Selective Z-Slicing**: Only process needed z-slices
   ```python
   z_slices=[10, 11, 12]  # Instead of all slices
   ```

2. **Use Plans for Large Data**: Create plans and process by color group
   ```python
   plan = quilt.create_extraction_plan(...)
   # Process color groups in parallel
   ```

3. **Memory-Mapped Backends**: For large files, use memory-mapped or zarr backends
   ```python
   # Only loads requested slices
   backend = ZarrBackend(zarr_array)
   ```

4. **Batch Processing**: Process multiple images together
   ```python
   # Process all images at once
   result = quilt.convert()  # (N, C', Y, X)
   ```

