# qlty

[![Python Version](https://img.shields.io/pypi/pyversions/qlty.svg)](https://pypi.org/project/qlty/)
[![License](https://img.shields.io/pypi/l/qlty.svg)](https://github.com/phzwart/qlty/blob/main/LICENSE)

**qlty** is a Python library designed to handle large 2D or 3D tensors efficiently by splitting them into smaller, manageable chunks. This library is particularly useful for processing large datasets that do not fit into memory, enabling chunked processing for machine learning workflows.

## Features

- **Efficient Tensor Splitting**: Split large tensors into overlapping patches for processing
- **Intelligent Stitching**: Reassemble patches with weighted averaging to handle overlaps
- **Border Handling**: Manage border pixels to minimize artifacts during stitching
- **Memory Management**: Support for both in-memory and disk-cached processing
- **2D and 3D Support**: Handle both 2D images and 3D volumes
- **2.5D Quilt**: Convert 3D volumetric data (N, C, Z, Y, X) to multi-channel 2D by slicing Z dimension into channels
- **Backend System**: Unified interface for multiple data sources (torch.Tensor, Zarr, HDF5, memory-mapped arrays)
- **Image Stack Utilities**: Convert image file stacks (TIFF, PNG) to efficient Zarr format with pattern matching
- **False Color Visualization**: UMAP-based false-color visualization of 2D images using patch-based dimensionality reduction
- **Sparse Data Handling**: Filter out patches with missing or invalid data

## Quick Start

### Installation

```bash
pip install qlty torch zarr numpy einops dask numba
```

### Basic Usage

```python
import torch
from qlty import NCYXQuilt

# Create a quilt object for 128x128 images
quilt = NCYXQuilt(
    Y=128, X=128,
    window=(32, 32),      # Patch size
    step=(16, 16),        # Step size (50% overlap)
    border=(5, 5),        # Border region to downweight
    border_weight=0.1     # Weight for border pixels
)

# Split data into patches
data = torch.randn(10, 3, 128, 128)  # (N, C, Y, X)
patches = quilt.unstitch(data)       # Returns (M, C, 32, 32)

# Process patches (e.g., with a neural network)
processed_patches = your_model(patches)

# Stitch back together
reconstructed, weights = quilt.stitch(processed_patches)
```

## Modules Overview

### 2.5D Quilt (New in 1.2.3)

**Convert 3D volumetric data to multi-channel 2D:**

- **`NCZYX25DQuilt`**: Converts 3D data (N, C, Z, Y, X) to 2.5D multi-channel data
  - Flexible channel specifications: extract specific z-slices or compute aggregations (mean, std)
  - Two accumulation modes: flatten to 2D planes or keep 3D structure
  - Works with multiple data sources: torch.Tensor, Zarr, HDF5, memory-mapped arrays
  - Selective z-slice processing and boundary handling modes
  - Direct integration with 2D patch pair extraction

```python
from qlty import NCZYX25DQuilt

data = torch.randn(5, 1, 20, 100, 100)  # (N, C, Z, Y, X)
spec = {'identity': [-1, 0, 1], 'mean': [[-2, -3], [2, 3]]}
quilt = NCZYX25DQuilt(data_source=data, channel_spec=spec, accumulation_mode="2d")
result = quilt.convert()  # Shape: (5*20, 5, 100, 100) - each z-slice becomes separate 2D image
```

### Backend System (New in 1.2.3)

**Unified interface for multiple data sources:**

- **`TensorLike3D`**: Makes any backend look like a PyTorch tensor
- **Backends**: `InMemoryBackend`, `ZarrBackend`, `HDF5Backend`, `MemoryMappedBackend`
- **Convenience functions**: `from_zarr()`, `from_hdf5()`, `from_memmap()`

```python
from qlty.backends_2_5D import from_zarr, from_hdf5

# Load from Zarr
data = from_zarr("data.zarr")

# Load from HDF5
data = from_hdf5("data.h5", "/images/stack")

# Use with 2.5D Quilt
quilt = NCZYX25DQuilt(data_source=data, channel_spec={'identity': [0]})
```

### Image Stack Utilities (New in 1.2.3)

**Convert image file stacks to Zarr format:**

- **`stack_files_to_zarr()`**: Automatically groups image files into 3D stacks
  - Pattern matching for flexible file naming
  - Automatic gap detection and warnings
  - Support for single-channel and multi-channel images
  - Customizable axis orders and chunk sizes

```python
from qlty.utils.stack_to_zarr import stack_files_to_zarr

result = stack_files_to_zarr(
    directory="/path/to/images",
    extension=".tif",
    pattern=r"(.+)_(\d+)\.tif$"  # Matches: stack_001.tif, stack_002.tif, etc.
)
# Returns metadata dict with zarr paths and stack information
```

### In-Memory Classes

**For datasets that fit in memory:**

- **`NCYXQuilt`**: 2D tensor splitting and stitching
  - Handles tensors of shape `(N, C, Y, X)`
  - Fast in-memory processing
  - Optional Numba acceleration for stitching

- **`NCZYXQuilt`**: 3D tensor splitting and stitching
  - Handles tensors of shape `(N, C, Z, Y, X)`
  - Same interface as 2D version

### Disk-Cached Classes

**For very large datasets that don't fit in memory:**

- **`LargeNCYXQuilt`**: 2D with disk caching
  - Uses Zarr for on-disk caching
  - Processes chunks incrementally
  - Supports mean and standard deviation computation

- **`LargeNCZYXQuilt`**: 3D with disk caching
  - Same features as 2D Large version
  - Handles 3D volumes efficiently

## Key Concepts

### Unstitching

Unstitching splits a large tensor into smaller, overlapping patches. The patches are created using a sliding window approach:

```python
# Window size: (32, 32) - each patch is 32x32 pixels
# Step size: (16, 16) - window moves 16 pixels each step
# This creates 50% overlap between patches
```

### Stitching

Stitching reassembles patches back into the original tensor shape. Overlapping regions are averaged using a weight matrix:

- **Center pixels**: Full weight (1.0)
- **Border pixels**: Reduced weight (default 0.1)
- **Result**: Smooth reconstruction without edge artifacts

### Border Handling

Border regions are pixels near the edges of each patch that may have lower confidence due to:
- Limited context
- Edge effects in neural networks
- Alignment artifacts

By downweighting borders, you get better overall results.

## Examples

### Example 1: 2D Image Processing

```python
import torch
from qlty import NCYXQuilt

# Create quilt object
quilt = NCYXQuilt(
    Y=256, X=256,
    window=(64, 64),
    step=(32, 32),
    border=(8, 8),
    border_weight=0.1
)

# Load your data
images = torch.randn(20, 3, 256, 256)  # 20 RGB images

# Split into patches
patches = quilt.unstitch(images)
print(f"Patches shape: {patches.shape}")  # (M, 3, 64, 64)

# Process with your model
output_patches = your_model(patches)

# Stitch back together
reconstructed, weights = quilt.stitch(output_patches)
print(f"Reconstructed shape: {reconstructed.shape}")  # (20, C, 256, 256)
```

### Example 2: Training Data Pairs

```python
from qlty import NCYXQuilt

quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16), border=(4, 4))

# Input and target tensors
input_data = torch.randn(10, 3, 128, 128)   # Input images
target_data = torch.randn(10, 128, 128)      # Target labels

# Unstitch both together
input_patches, target_patches = quilt.unstitch_data_pair(input_data, target_data)

# Train your model
for inp, tgt in zip(input_patches, target_patches):
    loss = criterion(model(inp), tgt)
    # ...
```

### Example 3: Large Dataset with Disk Caching

```python
from qlty import LargeNCYXQuilt
import tempfile
import os

# Create temporary identityory for cache
temp_dir = tempfile.mkdtemp()
filename = os.path.join(temp_dir, "my_data")

# Create Large quilt object
quilt = LargeNCYXQuilt(
    filename=filename,
    N=100,           # 100 images
    Y=512, X=512,   # Image size
    window=(128, 128),
    step=(64, 64),
    border=(10, 10),
    border_weight=0.1
)

# Process data in chunks
data = torch.randn(100, 3, 512, 512)

for i in range(quilt.N_chunks):
    index, patch = quilt.unstitch_next(data)

    # Process patch
    processed = your_model(patch.unsqueeze(0))

    # Accumulate result
    quilt.stitch(processed, index)

# Get final result
mean_result = quilt.return_mean()
std_result = quilt.return_mean(std=True)
```

### Example 4: Handling Missing Data

```python
from qlty import NCYXQuilt, weed_sparse_classification_training_pairs_2D

quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5))

# Data with missing labels (marked as -1)
input_data = torch.randn(10, 3, 128, 128)
labels = torch.ones(10, 128, 128) * (-1)  # All missing initially
labels[:, 20:108, 20:108] = 1.0            # Some valid data

# Unstitch with missing label handling
input_patches, label_patches = quilt.unstitch_data_pair(
    input_data, labels, missing_label=-1
)

# Filter out patches with no valid data
border_tensor = quilt.border_tensor()
valid_input, valid_labels, mask = weed_sparse_classification_training_pairs_2D(
    input_patches, label_patches, missing_label=-1, border_tensor=border_tensor
)

# Only valid patches remain
print(f"Valid patches: {valid_input.shape[0]}")
```

### Example 5: 3D Volume Processing

```python
from qlty import NCZYXQuilt

# Create 3D quilt
quilt = NCZYXQuilt(
    Z=64, Y=64, X=64,
    window=(32, 32, 32),
    step=(16, 16, 16),
    border=(4, 4, 4),
    border_weight=0.1
)

# 3D volume data
volume = torch.randn(5, 1, 64, 64, 64)  # (N, C, Z, Y, X)

# Process
patches = quilt.unstitch(volume)
processed = your_model(patches)
reconstructed, weights = quilt.stitch(processed)
```

## API Reference

### NCYXQuilt (2D In-Memory)

```python
NCYXQuilt(Y, X, window, step, border, border_weight=1.0)
```

**Parameters:**
- `Y` (int): Height of input tensors
- `X` (int): Width of input tensors
- `window` (tuple): Patch size `(Y_size, X_size)`
- `step` (tuple): Step size `(Y_step, X_step)`
- `border` (int, tuple, or None): Border size in pixels
- `border_weight` (float): Weight for border pixels (0.0 to 1.0)

**Methods:**
- `unstitch(tensor)`: Split tensor into patches
- `stitch(patches, use_numba=True)`: Reassemble patches
- `unstitch_data_pair(tensor_in, tensor_out, missing_label=None)`: Split input/output pairs
- `border_tensor()`: Get border mask tensor
- `get_times()`: Get number of patches per dimension

### NCZYXQuilt (3D In-Memory)

Same interface as `NCYXQuilt` but for 3D data:
- Input shape: `(N, C, Z, Y, X)`
- Window and step are 3-element tuples: `(Z, Y, X)`

### LargeNCYXQuilt (2D Disk-Cached)

```python
LargeNCYXQuilt(filename, N, Y, X, window, step, border, border_weight=0.1)
```

**Additional Parameters:**
- `filename` (str): Base filename for Zarr cache files
- `N` (int): Number of images in dataset

**Methods:**
- `unstitch_next(tensor)`: Get next patch (generator-like)
- `stitch(patch, index, patch_var=None)`: Accumulate patch
- `return_mean(std=False, normalize=False)`: Get final result
- `unstitch_and_clean_sparse_data_pair(...)`: Split and filter sparse data

### LargeNCZYXQuilt (3D Disk-Cached)

Same interface as `LargeNCYXQuilt` but for 3D volumes.

## Utility Functions

### Cleanup Functions

```python
from qlty import weed_sparse_classification_training_pairs_2D, weed_sparse_classification_training_pairs_3D

# Filter out patches with no valid data
valid_in, valid_out, mask = weed_sparse_classification_training_pairs_2D(
    tensor_in, tensor_out, missing_label, border_tensor
)
```

## Best Practices

1. **Choose Appropriate Overlap**:
   - 50% overlap (step = window/2) is common
   - More overlap = smoother results but slower processing

2. **Set Border Size**:
   - Typically 10-20% of window size
   - Larger borders for networks sensitive to edges

3. **Border Weight**:
   - 0.1 is a good default
   - 0.0 for complete exclusion, 1.0 for full weight

4. **Memory vs Disk**:
   - Use in-memory classes if data fits in RAM
   - Use Large classes for datasets > several GB

5. **Softmax Warning**:
   - Apply softmax AFTER stitching, not before
   - Averaging softmaxed tensors ≠ softmax of averaged tensors

## Performance Tips

- **Numba Acceleration**: `NCYXQuilt.stitch()` uses Numba by default for 2D
- **Batch Processing**: Process patches in batches for better GPU utilization
- **Zarr Chunking**: Large classes use optimized Zarr chunk sizes

## Dependencies

- **torch**: PyTorch tensors
- **numpy**: Numerical operations
- **zarr**: Disk caching (Large classes)
- **einops**: Tensor reshaping
- **dask**: Parallel processing (Large classes)
- **numba**: JIT compilation (optional, for 2D stitching)

## License

BSD License

## Citation

If you use qlty in your research, please cite:

```bibtex
@software{qlty,
  title = {qlty: Efficient Tensor Splitting and Stitching for Large Datasets},
  author = {Zwart, Petrus H.},
  year = {2024},
  url = {https://github.com/phzwart/qlty}
}
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## Support

- **Documentation**: https://qlty.readthedocs.io
- **Issues**: https://github.com/phzwart/qlty/issues
- **Email**: PHZwart@lbl.gov

## Changelog

See `HISTORY.rst` for version history and changes.
