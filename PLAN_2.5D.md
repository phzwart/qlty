# Plan for 2.5D Quilt Implementation

## Overview
Create a "2.5D" version that converts 3D data (N, C, Z, Y, X) into 2D multi-channel data (N, C', Y, X) by slicing the Z dimension into channels according to a flexible specification.

## Key Requirements

### 1. Channel Specification Format

**Supported Format (ZOperation Enum-based):**
```python
from enum import Enum

class ZOperation(Enum):
    DIRECT = 1  # Single pixel extraction
    MEAN = 3   # Mean of multiple pixels
    # Future: MAX = 5, MIN = 6, MEDIAN = 7

spec = {
    ZOperation.DIRECT: (-1, 0, 1),
    ZOperation.MEAN: ((-1, -2, -3), (1, 2, 3))
}
```

**Alternative Format (String keys, auto-converted to enum):**
```python
spec = {
    'direct': [-1, 0, 1],  # Direct indexing
    'mean': [[-1, -2, -3], [1, 2, 3]],  # Mean aggregations
    # Future: 'max': [[0, 1, 2]]
}
```

**Design Decision:**
- Only support ZOperation enum-based specs (no integer keys)
- String format is user-friendly and auto-converts to enum internally
- Enum format is explicit and type-safe
- Parser normalizes string keys to ZOperation enum

**Example Output:**
- Spec: `{ZOperation.DIRECT: (-1,0,1), ZOperation.MEAN: ((-1,-2,-3), (1,2,3))}`
- Produces 5 channels per input channel:
  - Channel 0: z0-1
  - Channel 1: z0
  - Channel 2: z0+1
  - Channel 3: mean(z0-1, z0-2, z0-3)
  - Channel 4: mean(z0+1, z0+2, z0+3)

### 2. Multichannel Input Handling
- For input with C channels, apply the same Z-slicing to each input channel
- Output channels = C_input × C_z_slices
- Example: 3 input channels × 5 Z-slices = 15 output channels

### 2.5. Accumulation Modes

**Two modes for organizing 2.5D results:**

#### Mode 1: "2d" - 2D Planes (Default)
- Flattens Z dimension into channels
- Output shape: `(N, C', Y, X)` where `C' = C_in × C_z_slices`
- Each z-slice becomes a set of channels
- Can be directly fed to 2D quilt operations
- **Use case**: When you want to treat each z-slice as independent 2D images

#### Mode 2: "3d" - 3D Stack
- Keeps Z dimension separate
- Output shape: `(N, C', Z, Y, X)` where `C' = C_in × C_z_slices`
- Maintains 3D structure with expanded channels
- **Use case**: When you need to preserve spatial relationships in Z dimension

**Example:**
```python
# Input: (1, 1, 10, 100, 100) - 1 image, 1 channel, 10 z-slices
# Channel spec produces 5 channels per input channel

# Mode "2d":
# Output: (1, 5, 100, 100) - flattened, ready for 2D processing

# Mode "3d":
# Output: (1, 5, 10, 100, 100) - maintains 3D structure
```

### 3. Boundary Handling

Need to handle Z-boundary conditions when offsets go outside [0, Z-1] range.

**Supported Modes:**
- `"clamp"`: Repeat edge slices (default, most common)
- `"zero"`: Zero-padding for out-of-bounds slices
- `"reflect"`: Mirror padding (maintains continuity, often better than clamp)
- `"skip"`: Skip invalid slices (reduces channel count at boundaries - use with caution)

**Design Decision:**
- Default to `"clamp"` for simplicity
- `"reflect"` recommended for image processing applications
- Make configurable via `boundary_mode` parameter
- Apply consistently across all operations in a spec

### 4. Flexible 3D Data Source Interface

**Key Design Principle:**
- **Externally**: Data source can be anything (zarr, HDF5, tiled server, file object, etc.)
- **Internally in qlty**: Must behave like a PyTorch tensor (indexable, returns tensors)
- **Memory concern**: Never load entire dataset into memory - load on-demand via slicing
- **Tensor-like interface**: Support indexing like `data[n, c, z, y, x]` or `data[n, c, z_range]` that returns `torch.Tensor`

**Proposed Architecture:**
```python
class TensorLike3D:
    """
    Wrapper that makes any data source look like a PyTorch tensor.
    Supports tensor-like indexing but loads data on-demand.
    """
    
    def __init__(self, backend: DataSource3DBackend):
        """
        Parameters
        ----------
        backend : DataSource3DBackend
            Backend that actually stores/loads the data
        """
        self.backend = backend
        self._shape = backend.get_shape()  # (N, C, Z, Y, X)
        self._dtype = backend.get_dtype()
    
    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        """Return (N, C, Z, Y, X) shape - like tensor.shape"""
        return self._shape
    
    @property
    def dtype(self) -> torch.dtype:
        """Return dtype - like tensor.dtype"""
        return self._dtype
    
    def __getitem__(self, key) -> torch.Tensor:
        """
        Tensor-like indexing that returns PyTorch tensors.
        Loads data on-demand from backend.
        
        Examples:
        - data[0] -> (C, Z, Y, X) tensor
        - data[0, 1] -> (Z, Y, X) tensor  
        - data[0, 1, 5:10] -> (5, Y, X) tensor
        - data[0, 1, 5] -> (Y, X) tensor
        """
        # Parse key and load from backend
        # Convert to torch.Tensor before returning
        pass


class DataSource3DBackend(ABC):
    """
    Backend interface for actual data storage/loading.
    Implementations handle the specifics of different data sources.
    """
    
    @abstractmethod
    def get_shape(self) -> Tuple[int, int, int, int, int]:
        """Return (N, C, Z, Y, X) shape"""
        pass
    
    @abstractmethod
    def get_dtype(self) -> torch.dtype:
        """Return data type (as torch.dtype)"""
        pass
    
    @abstractmethod
    def load_slice(
        self, 
        n: Optional[int] = None,
        c: Optional[int] = None, 
        z: Optional[Union[int, slice]] = None,
        y: Optional[Union[int, slice]] = None,
        x: Optional[Union[int, slice]] = None,
    ) -> torch.Tensor:
        """
        Load data slice and return as torch.Tensor.
        Loads only what's requested - never entire dataset.
        
        Parameters
        ----------
        n, c, z, y, x : int, slice, or None
            Indices/slices for each dimension. None means all.
        
        Returns
        -------
        torch.Tensor
            Requested slice as PyTorch tensor
        """
        pass
    
    @property
    @abstractmethod
    def supports_batch_loading(self) -> bool:
        """Whether backend can efficiently load multiple z-slices at once"""
        pass
```

**Backend Implementations:**
- `InMemoryBackend`: Wraps existing `torch.Tensor` (no-op, just returns views)
- `ZarrBackend`: Loads from zarr arrays on-demand
- `HDF5Backend`: Loads from HDF5 files on-demand
- `TiffStackBackend`: Loads from TIFF stacks on-demand
- `TiledServerBackend`: Loads from tiled servers (neuroglancer, etc.) on-demand

**Usage Pattern:**
```python
# User provides any backend
zarr_backend = ZarrBackend("data.zarr")
data = TensorLike3D(zarr_backend)

# Now use it like a tensor in qlty
# qlty code doesn't know it's zarr - it's just a tensor-like object
quilt = NCZYX25DQuilt(data, channel_spec={...})

# When qlty does data[0, 1, 5:10], it gets a torch.Tensor
# Backend only loads those specific slices
```

### 5. Architecture Design

**Key Requirements:**
1. **Selective Slicing**: Specify which slices to extract (not process everything)
2. **Plan System**: Generate extraction/stitching schedule for parallelization
3. **Coloring Groups**: Support color-based parallelization (like qlty2D colored stitching)
4. **Two Accumulation Modes**: 3D stack or 2D planes

#### Core Classes

**`NCZYX25DQuilt`**: Main class for 2.5D operations
- Converts 3D→2.5D with channel specification
- Supports selective slicing (specify which z-slices to process)
- Generates extraction/stitching plans for parallelization
- Two output modes: 3D stack or 2D planes

**`ExtractionPlan`**: Specifies what to extract and when
- Lists all patches to extract with their indices
- Includes color group assignments for parallelization
- Can be serialized/deserialized for distributed processing

**`StitchingPlan`**: Specifies how to stitch back together
- Maps patch indices to output locations
- Includes color group information
- Coordinates accumulation order

### 6. Class Structure

```python
class NCZYX25DQuilt:
    """
    Converts 3D data (N, C, Z, Y, X) to 2.5D multi-channel data.
    
    Supports selective slicing, plan generation for parallelization,
    and two accumulation modes (3D stack or 2D planes).
    
    Works with tensor-like objects that support indexing and return torch.Tensor.
    """
    
    def __init__(
        self,
        data_source: Union[torch.Tensor, TensorLike3D, str],
        channel_spec: Union[
            Dict[str, Union[List[int], List[List[int]]]],
            Dict[ZOperation, Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]],
        ],
        boundary_mode: str = "clamp",
        accumulation_mode: str = "2d",  # "2d" or "3d"
        z_slices: Optional[Union[slice, List[int]]] = None,  # Selective slicing
        window: Optional[Tuple[int, int]] = None,  # For 2D quilting
        step: Optional[Tuple[int, int]] = None,  # For 2D quilting
        border: Optional[Union[int, Tuple[int, int]]] = None,
        border_weight: float = 0.1,
        device: str = "cpu",
        group_by_operation: bool = False,
    ):
        """
        Parameters
        ----------
        data_source : Union[torch.Tensor, TensorLike3D, str]
            - torch.Tensor: In-memory 3D data (N, C, Z, Y, X)
            - TensorLike3D: Tensor-like wrapper around any backend
            - str: Path to zarr file - automatically wrapped
        channel_spec : Dict
            Channel specification with ZOperation enum or string keys
        boundary_mode : str
            Z-boundary handling: "clamp", "zero", "reflect", "skip"
        accumulation_mode : str
            "2d": Flatten to (N, C', Y, X) - 2D planes
            "3d": Keep as (N, C', Z, Y, X) - 3D stack
        z_slices : Optional[Union[slice, List[int]]]
            Which z-slices to process. None = all slices.
            Example: slice(0, 100, 2) or [0, 5, 10, 15]
        window : Optional[Tuple[int, int]]
            Window size for 2D quilting (Y, X). Required if accumulation_mode="2d"
        step : Optional[Tuple[int, int]]
            Step size for 2D quilting (Y, X). Required if accumulation_mode="2d"
        border : Optional[Union[int, Tuple[int, int]]]
            Border size for 2D quilting
        border_weight : float
            Border weight for stitching
        device : str
            Device for computation: "cpu", "cuda", "mps"
        group_by_operation : bool
            If True, group output channels by operation type
        """
    
    def create_extraction_plan(
        self,
        color_y_mod: Optional[int] = None,
        color_x_mod: Optional[int] = None,
    ) -> ExtractionPlan:
        """
        Create extraction plan specifying what to extract and when.
        Includes color group assignments for parallelization.
        
        Parameters
        ----------
        color_y_mod : Optional[int]
            Color modulo for Y dimension (for parallelization)
        color_x_mod : Optional[int]
            Color modulo for X dimension (for parallelization)
        
        Returns
        -------
        ExtractionPlan
            Plan with patch indices, z-slices, and color groups
        """
        pass
    
    def create_stitching_plan(self) -> StitchingPlan:
        """
        Create stitching plan specifying how to stitch back together.
        Coordinates accumulation order and color groups.
        
        Returns
        -------
        StitchingPlan
            Plan with output locations and accumulation order
        """
        pass
    
    def get_channel_metadata(self) -> List[Dict]:
        """
        Returns metadata for each output channel.
        
        Returns
        -------
        List[Dict]
            [{'operation': 'direct', 'offsets': (-1,), 'input_channel': 0, 'z0': 0},
             {'operation': 'mean', 'offsets': (-1,-2,-3), 'input_channel': 0, 'z0': 0}, ...]
        """
        pass
    
    def validate_spec(self) -> Tuple[bool, List[str]]:
        """
        Validate channel specification against data source.
        
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_warnings_or_errors)
        """
        pass
    
    @classmethod
    def from_zarr(cls, zarr_path: str, channel_spec: Dict, **kwargs):
        """
        Convenience constructor for zarr files.
        Creates TensorLike3D with ZarrBackend automatically.
        """
        from qlty.backends import ZarrBackend, TensorLike3D
        backend = ZarrBackend(zarr_path)
        data = TensorLike3D(backend)
        return cls(data, channel_spec, **kwargs)
    
    @classmethod
    def from_tiff_stack(cls, tiff_path: str, channel_spec: Dict, **kwargs):
        """
        Convenience constructor for TIFF stacks.
        Creates TensorLike3D with TiffStackBackend automatically.
        """
        from qlty.backends import TiffStackBackend, TensorLike3D
        backend = TiffStackBackend(tiff_path)
        data = TensorLike3D(backend)
        return cls(data, channel_spec, **kwargs)
    
    def to_ncyx_quilt(self, **quilt_kwargs) -> NCYXQuilt:
        """
        Convert to 2D and create NCYXQuilt in one step.
        Returns configured NCYXQuilt with converted data.
        """
        pass
```

### 7. Plan System for Parallelization

**Problem**: Large classes not suitable for parallelization. Need explicit plans.

**Solution**: Generate extraction and stitching plans that specify:
- What to extract and when
- How to stitch back together and when
- Color group assignments for safe parallelization

```python
@dataclass
class PatchExtraction:
    """Single patch extraction specification"""
    patch_idx: int  # Linear patch index
    n: int  # Image index
    z0: int  # Center z-slice
    y_start: int  # Y start (if 2D quilting)
    y_stop: int  # Y stop
    x_start: int  # X start (if 2D quilting)
    x_stop: int  # X stop
    color_y_idx: int  # Color group for Y
    color_x_idx: int  # Color group for X
    required_z_indices: List[int]  # All z-slices needed for this patch


@dataclass
class ExtractionPlan:
    """Plan for extracting patches"""
    patches: List[PatchExtraction]
    color_groups: Dict[Tuple[int, int], List[int]]  # (color_y, color_x) -> patch indices
    total_patches: int
    
    def get_patches_for_color(self, color_y: int, color_x: int) -> List[PatchExtraction]:
        """Get all patches for a specific color group"""
        pass
    
    def serialize(self) -> Dict:
        """Serialize plan for distributed processing"""
        pass
    
    @classmethod
    def deserialize(cls, data: Dict) -> 'ExtractionPlan':
        """Deserialize plan"""
        pass


@dataclass
class StitchingPlan:
    """Plan for stitching patches back together"""
    output_shape: Tuple[int, ...]  # (N, C', Y, X) or (N, C', Z, Y, X)
    patch_mappings: Dict[int, Dict]  # patch_idx -> {output_location, weight, ...}
    color_groups: Dict[Tuple[int, int], List[int]]  # Same as extraction plan
    
    def get_stitch_order(self, color_y: int, color_x: int) -> List[int]:
        """Get patch indices to stitch for a color group"""
        pass
    
    def serialize(self) -> Dict:
        """Serialize plan for distributed processing"""
        pass
    
    @classmethod
    def deserialize(cls, data: Dict) -> 'StitchingPlan':
        """Deserialize plan"""
        pass
```

**Usage Pattern for Parallelization:**
```python
# Create plans once
extraction_plan = quilt.create_extraction_plan(color_y_mod=4, color_x_mod=4)
stitching_plan = quilt.create_stitching_plan()

# Distribute plans to workers
# Each worker processes one color group
for color_y in range(color_y_mod):
    for color_x in range(color_x_mod):
        patches = extraction_plan.get_patches_for_color(color_y, color_x)
        # Process patches in parallel (no race conditions)
        results = process_patches(patches)
        # Stitch according to plan
        stitching_plan.stitch_color_group(color_y, color_x, results)
```

### 8. Channel Specification Parser

```python
def parse_channel_spec(
    spec: Union[
        Dict[str, Union[List[int], List[List[int]]]],
        Dict[ZOperation, Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]],
    ]
) -> List[ChannelOperation]:
    """
    Parse channel specification into list of operations.
    Supports string keys (auto-converted to enum) or ZOperation enum keys.
    Normalizes to internal ZOperation enum representation.
    
    Parameters
    ----------
    spec : Dict
        Channel specification with string or ZOperation enum keys
        
    Returns
    -------
    List[ChannelOperation]
        Ordered list of channel operations to apply
        
    Raises
    ------
    ValueError
        If spec contains invalid keys (not string or ZOperation enum)
    """
    
@dataclass
class ChannelOperation:
    op_type: ZOperation  # Use enum instead of int
    offsets: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
    output_channels: int  # Number of channels this operation produces
    name: Optional[str] = None  # Optional name for this operation
    
    def get_required_z_range(self, z0: int) -> Tuple[int, int]:
        """
        Get [z_min, z_max] needed to compute this operation at z0.
        Useful for optimizing data loading.
        """
        if self.op_type == ZOperation.DIRECT:
            offsets = self.offsets if isinstance(self.offsets[0], int) else self.offsets[0]
            z_indices = [z0 + offset for offset in offsets]
            return (min(z_indices), max(z_indices) + 1)
        elif self.op_type == ZOperation.MEAN:
            all_offsets = []
            for offset_group in self.offsets:
                if isinstance(offset_group, (list, tuple)):
                    all_offsets.extend(offset_group)
                else:
                    all_offsets.append(offset_group)
            z_indices = [z0 + offset for offset in all_offsets]
            return (min(z_indices), max(z_indices) + 1)
        return (z0, z0 + 1)
```

### 9. Processing Flow

#### 8.1 Plan-Based Processing (Recommended for Parallelization)

1. **Create Plans**:
   ```python
   extraction_plan = quilt.create_extraction_plan(color_y_mod=4, color_x_mod=4)
   stitching_plan = quilt.create_stitching_plan()
   ```

2. **Extract Patches** (can be parallelized by color group):
   - For each patch in extraction_plan:
     - Load required z-slices: `data[n, c, z_indices]` → `torch.Tensor`
     - Apply channel operations → (C', Y, X) or (C', window[0], window[1])
     - Store patch with its index and color group

3. **Process Patches** (parallel by color group):
   - Each color group processes independently
   - No race conditions since color groups don't overlap

4. **Stitch Results** (according to stitching_plan):
   - Accumulate patches back into final result
   - Handle overlapping regions with weights
   - Result depends on accumulation_mode:
     - "2d": (N, C', Y, X)
     - "3d": (N, C', Z, Y, X)

#### 8.2 Direct Processing (Simple, Sequential)

1. **Initialization**:
   - Parse channel specification
   - Validate data source
   - Compute output channel count: `C_out = C_in × sum(operation.output_channels)`
   - Determine z-slices to process (from `z_slices` parameter)

2. **Conversion Process** (Memory-Efficient):
   - For each image n in [0, N):
     - For each z-slice z0 in selected z_slices:
       - **Load only needed z-slices on-demand**: `data[n, c, z_indices]` → `torch.Tensor`
       - Apply channel operations to create output channels
       - If accumulation_mode="2d": Apply 2D quilting (window/step) → patches
       - If accumulation_mode="3d": Stack into (C', Y, X) for this z-slice
     - Accumulate according to mode:
       - "2d": Stitch patches → (C', Y, X)
       - "3d": Stack z-slices → (C', Z, Y, X)
   - Result: (N, C', Y, X) or (N, C', Z, Y, X)
   - **Key**: Never load entire volume - only requested z-slices via tensor-like indexing

#### 8.3 Integration with 2D Quilt

- If accumulation_mode="2d": Can use existing `NCYXQuilt` for stitching
- If accumulation_mode="3d": Results are already in 3D format
- Plans enable distributed/parallel processing

### 10. Memory Management

**Core Principle**: Never load entire dataset into memory. Load only what's needed on-demand.

#### Tensor-Like Interface:
- `TensorLike3D` supports tensor indexing: `data[n, c, z]` → returns `torch.Tensor`
- Backend only loads requested slices when indexed
- qlty code treats it like a normal tensor - no special handling needed

#### For In-Memory Data (torch.Tensor):
- Direct indexing - no overhead
- Process slice-by-slice to minimize memory footprint
- Optionally cache converted 2.5D data if it fits in memory

#### For Large Data (Zarr, HDF5, etc.):
- Backend loads only requested z-slices on-demand
- Use chunked processing (process one image at a time)
- Optionally write intermediate 2.5D result to disk (zarr)
- Support streaming: convert on-the-fly as 2D quilt requests patches
- **Key**: Indexing `data[n, c, z_range]` triggers backend to load only that range

#### Memory Efficiency Strategy:
1. **Lazy Loading**: Data loaded only when indexed
2. **Slice-Level Granularity**: Load individual z-slices, not entire volumes
3. **Batch Optimization**: If backend supports it, batch load multiple z-slices efficiently
4. **Cache Management**: Optional LRU cache for recently accessed slices

### 11. Implementation Steps

**Phase 0: Design Validation** (NEW - Critical First Step)
- Implement minimal prototype with core logic
- Test with real data to validate design decisions
- Iterate on channel spec syntax based on usability
- Validate boundary modes with actual use cases
- Get user feedback on API design

1. **Phase 1: Core Conversion Logic**
   - Implement channel specification parser (support enum and string formats)
   - Implement Z-slicing logic with all boundary modes
   - Implement two accumulation modes (2D planes and 3D stack)
   - Create basic `NCZYX25DQuilt` class for in-memory data
   - Implement selective slicing (z_slices parameter)
   - Implement `get_channel_metadata()` method
   - Implement `validate_spec()` method
   - Unit tests for channel spec parsing and conversion
   - Edge case testing

2. **Phase 2: Plan System & Parallelization Support**
   - Implement `ExtractionPlan` class
   - Implement `StitchingPlan` class
   - Implement `create_extraction_plan()` with color group support
   - Implement `create_stitching_plan()` with accumulation coordination
   - Add plan serialization/deserialization for distributed processing
   - Unit tests for plan generation
   - Unit tests for color group assignments
   - Verify no race conditions in parallel processing

3. **Phase 3: Tensor-Like Interface & Backend Implementations**
   - Define `TensorLike3D` class (tensor-like wrapper)
   - Define `DataSource3DBackend` interface
   - Implement `InMemoryBackend` (wraps torch.Tensor - no-op)
   - Implement `ZarrBackend` (loads from zarr on-demand)
   - Implement chunked processing for large datasets
   - Add optional disk caching for converted 2.5D data
   - Ensure all backends return `torch.Tensor` from indexing
   - Unit tests for tensor-like behavior
   - Unit tests for different backends
   - Integration tests with large data (verify memory efficiency)

4. **Phase 4: Integration & Convenience**
   - Convenience methods (`from_zarr`, `from_tiff_stack`, `to_ncyx_quilt`)
   - Performance optimization (batch loading, parallel processing)
   - Device support (CPU/CUDA/MPS)
   - Integration with existing 2D quilt classes
   - Documentation and examples

5. **Phase 5: Polish & Optimization**
   - Comprehensive documentation (quick start, visual diagrams, comparison guide)
   - Performance tuning guide (including parallelization strategies)
   - Migration guide from 3D quilting
   - Benchmark suite (including parallel performance)
   - Final optimization pass

### 12. Edge Cases to Consider

1. **Empty channel spec**: `{}` → should raise ValueError with helpful message
2. **Invalid offsets**: Offsets that go way beyond Z bounds → warn if > ±Z, error if extreme
3. **Negative offsets only**: `{1: (-5, -4, -3)}` - all before z0 → handle with boundary mode
4. **Positive offsets only**: `{1: (1, 2, 3)}` - all after z0 → handle with boundary mode
5. **Overlapping operations**: Same offsets in different operations → allow (duplicate channels)
6. **Zero offsets**: `{1: (0,)}` - just current slice → valid, produces single channel
7. **Single channel input**: C=1 should work the same → no special handling needed
8. **Single slice Z**: Z=1 - what happens? → all offsets map to z0, boundary mode applies
9. **Z=0 case**: What if Z dimension is missing entirely (pure 2D data)? → raise error, suggest 2D quilt
10. **Asymmetric offsets**: All offsets on one side (`{1: (-5,-4,-3,-2,-1)}`) → valid, handle with boundary
11. **Large offset gaps**: `{1: (-100, 0, 100)}` - should warn if offsets span > Z
12. **Duplicate offsets**: `{1: (-1, -1, 0)}` - allow (redundant but valid)
13. **Empty offset groups**: `{3: ((), (1,2))}` - raise error for empty tuple
14. **Mixed offset types**: Validate that all offsets in a group are consistent

### 13. API Design Questions

1. **Should conversion be lazy or eager?**
   - Lazy: Convert on-demand as patches are requested
   - Eager: Convert entire dataset upfront
   - **Recommendation**: Support both, default to lazy for large data

2. **Should we provide a combined class?**
   - `NCZYX25DQuilt` + `NCYXQuilt` = `NCZYX25DQuilt2D`?
   - Or keep separate and provide convenience wrapper?

3. **How to handle the "Large" version?**
   - `LargeNCZYX25DQuilt` that uses disk caching?
   - Or reuse `NCZYX25DQuilt` with appropriate data source?

### 14. Testing Strategy

1. **Unit Tests**:
   - Channel spec parsing (all formats: enum, string, int)
   - Invalid spec handling (empty, malformed, etc.)
   - Z-slicing logic (all boundary modes: clamp, zero, reflect, skip)
   - Channel counting (verify C_out calculation)
   - Channel metadata generation
   - Validation method (warnings and errors)
   - Data source implementations (all types)
   - Edge cases (Z=1, empty spec, extreme offsets, etc.)

2. **Integration Tests**:
   - Full pipeline: 3D → 2.5D → 2D patches → stitch
   - Plan-based processing: Extraction → Process → Stitching
   - Parallel processing: Verify color groups work correctly, no race conditions
   - Roundtrip tests: Convert 3D→2.5D→patches→stitch, compare with direct 3D processing
   - Consistency tests: Same result with different data sources (memory vs zarr)
   - Accumulation mode tests: Verify both 2D and 3D modes produce correct results
   - Selective slicing tests: Verify z_slices parameter works correctly
   - Large data with disk caching
   - Different data sources (tensor, zarr, file objects)
   - Convenience methods (`from_zarr`, `to_ncyx_quilt`)

3. **Performance Tests**:
   - Memory usage with large datasets
   - Speed comparison: in-memory vs disk-cached
   - Comparison with direct 3D processing
   - Batch loading optimization effectiveness
   - Device performance (CPU vs CUDA vs MPS)
   - Benchmark suite for regression testing

4. **Numerical Precision Tests**:
   - Ensure mean operations don't lose precision
   - Verify boundary modes produce expected results
   - Test with different dtypes (float32, float64, int types)

### 15. Open Questions - Resolved

1. **Channel ordering**: Should channels be ordered as specified, or grouped by operation type?
   - **Decision**: Order as specified by default, but provide optional `group_by_operation=False` parameter
   - Allows flexibility for different use cases

2. **Mean operation**: Should we support other operations (max, min, median)?
   - **Decision**: Design enum to be extensible, but start with DIRECT and MEAN only
   - Future operations: MAX, MIN, MEDIAN can be added as needed
   - Enum design allows easy extension without breaking changes

3. **Validation**: How strict should channel spec validation be?
   - **Decision**: Validate that requested slices are within reasonable bounds (e.g., ±Z)
   - Warn for offsets > ±Z but < ±2Z
   - Error for extreme offsets (> ±2Z) or malformed specs
   - Provide helpful error messages with suggestions

4. **Documentation**: Should channel spec support named channels?
   - **Decision**: Support string keys (auto-converted to enum) or ZOperation enum keys
   - String format is most user-friendly: `{'direct': [-1,0,1], 'mean': ...}`
   - Enum format is most explicit: `{ZOperation.DIRECT: (-1,0,1), ...}`
   - No integer format support - only ZOperation enum-based specs

## Documentation Priorities

1. **Quick Start Examples**
   - Common use cases (cryo-EM, confocal z-stacks, etc.)
   - Simple 3-line examples for each use case
   - Copy-paste ready code snippets

2. **Visual Diagrams**
   - Channel specification effects (before/after visualization)
   - Z-slicing operation diagrams
   - Boundary mode illustrations

3. **Comparison Guide**
   - When to use 2.5D vs full 3D quilting
   - Performance trade-offs
   - Memory considerations

4. **Performance Tuning Guide**
   - Optimizing for large datasets
   - Choosing appropriate data sources
   - Device selection (CPU/CUDA/MPS)
   - Batch loading strategies

5. **Migration Guide**
   - From existing 3D quilting code
   - From manual Z-slicing approaches
   - Best practices for conversion

6. **API Reference**
   - Complete function signatures
   - Parameter descriptions
   - Return value documentation
   - Examples for each method

## Critical Success Factors

1. **Usability**: Channel spec must be intuitive - consider user testing with real scientific imaging scenarios
2. **Performance**: Profile early with realistic data sizes (typical cryo-EM, confocal volumes)
3. **Debugging**: Excellent error messages when things go wrong, with actionable suggestions
4. **Documentation**: Examples for common scientific imaging scenarios (cryo-EM, confocal z-stacks, light sheet, etc.)
5. **Extensibility**: Design for future operations (max, min, median) without breaking changes

## Next Steps

1. ✅ Review and refine this plan (DONE - incorporated feedback)
2. ✅ Decide on open questions (DONE - all resolved)
3. **Phase 0**: Create minimal prototype and validate design with real data
4. Create detailed function signatures
5. Implement Phase 1 (core conversion logic)
6. Iterate based on testing and feedback

