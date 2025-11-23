# Plan: Stack Image Files to Zarr Utility

## Overview
Create a utility function to scan a directory for image files matching a pattern, group them into 3D stacks, and save each stack as a zarr file.

## Requirements

### Input
- **Directory path**: Root directory to scan
- **File extension**: e.g., `.tif`, `.tiff`, `.png`, `.jpg`
- **Pattern mask**: Regex pattern to extract (basename, counter, extension) from filenames
  - Example patterns:
    - `r"(.+)_(\d+)\.tif$"` → `image_001.tif` → basename="image_", counter=001
    - `r"(.+?)(\d+)\.tif$"` → `image001.tif` → basename="image", counter=001
    - `r"(.+)_z(\d+)\.tif$"` → `stack_z001.tif` → basename="stack_", counter=001

### Output
- **Zarr files**: One zarr file per unique stack (basename)
- **Stack metadata**: Information about each stack (dimensions, file list, etc.)

## Design

### Function Signature
```python
def stack_files_to_zarr(
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
) -> dict[str, dict]:
    """
    Scan directory for image files, group into 3D stacks, and save as zarr.

    Parameters
    ----------
    directory : str | Path
        Directory to scan for image files (top level only, non-recursive)
    extension : str
        File extension to match (e.g., '.tif', '.png')
    pattern : str | re.Pattern
        Regex pattern with two groups: (basename, counter)
        Example: r"(.+)_(\d+)\.tif$"
    output_dir : str | Path | None
        Directory to save zarr files. If None, saves in same directory.
    zarr_chunks : tuple[int, ...] | None
        Chunk size for zarr arrays. If None, uses reasonable defaults.
    dtype : np.dtype | None
        Data type for zarr arrays. If None, infers from first image.
    axis_order : str
        Axis order for multi-channel images. Default: "ZCYX"
        Options: "ZCYX", "CZYX", "ZYCX", etc.
        Single channel images always use "ZYX" regardless of this setting.
    output_naming : Callable[[str], str] | None
        Function to generate output zarr filename from basename.
        If None, uses default: f"{basename}.zarr"
        Example: lambda b: f"{b}_stack.zarr"
    sort_by_counter : bool
        Whether to sort files by counter value (default: True)
    dry_run : bool
        If True, only analyze files without creating zarr (default: False)

    Returns
    -------
    dict[str, dict]
        Dictionary mapping stack basename to metadata:
        {
            "stack_name": {
                "zarr_path": str,
                "shape": tuple[int, ...],  # (Z, C, Y, X) or (Z, Y, X)
                "dtype": np.dtype,
                "file_count": int,
                "files": list[str],  # Sorted list of file paths
                "counter_range": tuple[int, int],  # (min, max)
                "axis_order": str,  # Actual axis order used
            }
        }
    """
```

### Implementation Steps

#### Step 1: File Discovery and Parsing
1. Scan directory recursively (or non-recursively) for files with given extension
2. For each file, apply regex pattern to extract:
   - `basename`: Base name of the stack
   - `counter`: Numeric counter (z-index)
   - `extension`: File extension
3. Validate that pattern matches and counter can be parsed as integer
4. Group files by `basename` to form stacks

#### Step 2: Stack Analysis
For each unique basename (stack):
1. Sort files by counter value (if `sort_by_counter=True`)
2. Load first image to determine:
   - Image dimensions (Y, X)
   - Data type (dtype)
   - Number of channels (if applicable)
3. Determine stack dimensions:
   - Z = number of files in stack
   - Y, X = dimensions from first image
   - C = number of channels (if multi-channel images)
4. Validate all images in stack have same dimensions
5. Extract counter range (min, max)
6. Determine final shape based on `axis_order`:
   - Single channel: always `(Z, Y, X)` regardless of axis_order
   - Multi-channel: apply `axis_order` (default: `(Z, C, Y, X)`)

#### Step 3: Zarr Creation
For each stack:
1. Determine output path:
   - Use `output_naming(basename)` if provided, else `f"{basename}.zarr"`
   - If `output_dir` provided: `{output_dir}/{zarr_name}`
   - Otherwise: `{directory}/{zarr_name}`
2. Create zarr array with shape based on `axis_order`:
   - Single channel: always `(Z, Y, X)`
   - Multi-channel: apply `axis_order` (default: `(Z, C, Y, X)`)
3. Set chunk size:
   - Default: `(1, Y, X)` for single channel or based on axis_order for multi-channel
   - Or use `zarr_chunks` if provided
4. Load and write images sequentially:
   - For each file in sorted order:
     - Load image using `tifffile` for TIFF (best developed), fallback to PIL/imageio
     - Reshape/reorder according to `axis_order` if multi-channel
     - Write to zarr array at appropriate z-index
5. Save metadata as zarr attributes:
   - Store file list, counter range, axis_order, original pattern, etc.

#### Step 4: Error Handling
- Handle missing files in sequence (gaps in counter)
- Handle dimension mismatches
- Handle unsupported image formats
- Handle memory issues for large stacks

### Dependencies
- `zarr`: For creating zarr arrays
- `numpy`: For array operations
- `tifffile`: Primary library for TIFF files (best developed for scientific imaging)
- `PIL/Pillow`: Fallback for non-TIFF formats
- `imageio`: Additional fallback for various formats

### Example Usage

```python
from qlty.utils.stack_to_zarr import stack_files_to_zarr
import re

# Example 1: Simple pattern
result = stack_files_to_zarr(
    directory="/path/to/images",
    extension=".tif",
    pattern=r"(.+)_(\d+)\.tif$",
    output_dir="/path/to/zarr_output"
)

# Example 2: Custom pattern with axis order
result = stack_files_to_zarr(
    directory="/path/to/images",
    extension=".tiff",
    pattern=r"image_z(\d+)\.tiff$",  # basename="image_z", counter from group 1
    zarr_chunks=(1, 1, 256, 256),  # (Z, C, Y, X)
    dtype=np.float32,
    axis_order="ZCYX"  # Default, can change to "CZYX" etc.
)

# Example 3: Custom output naming
result = stack_files_to_zarr(
    directory="/path/to/images",
    extension=".tif",
    pattern=r"(.+)_(\d+)\.tif$",
    output_naming=lambda basename: f"{basename}_processed.zarr"
)

# Example 4: Dry run to analyze
result = stack_files_to_zarr(
    directory="/path/to/images",
    extension=".tif",
    pattern=r"(.+)_(\d+)\.tif$",
    dry_run=True  # Just analyze, don't create zarr
)

# Access results
for stack_name, metadata in result.items():
    print(f"Stack: {stack_name}")
    print(f"  Shape: {metadata['shape']}")
    print(f"  Files: {metadata['file_count']}")
    print(f"  Zarr: {metadata['zarr_path']}")
```

### File Structure
```
qlty/utils/
  stack_to_zarr.py  # Main utility function
  __init__.py       # Export function
```

### Testing Strategy
1. Create test directory with sample image files
2. Test pattern matching with various patterns
3. Test stack grouping and sorting
4. Test zarr creation and validation
5. Test error cases (missing files, dimension mismatches)
6. Test with different image formats
7. Test dry_run mode

### Implementation Notes
- Use `tifffile` as primary image loader (best developed for scientific imaging)
- Default axis order: "ZCYX" for multi-channel images
- Single-channel images always use "ZYX" regardless of axis_order setting
- Metadata stored as zarr attributes (file list, counter range, axis_order, etc.)
- Top-level directory scanning only (non-recursive)

### Future Enhancements
- Parallel loading/writing for large stacks
- Progress bar for long operations
- Support for different zarr compression options
- Metadata preservation (EXIF, etc.)
- Support for subdirectories (recursive scanning)
