# QLTY Codebase Review and Cleanup Proposal

## Executive Summary

This document provides a comprehensive review of the qlty codebase and proposes a structured cleanup and rework plan. The codebase handles 2D/3D tensor splitting and stitching for large datasets, but suffers from significant code duplication, inconsistent implementations, and architectural issues.

## Current State Analysis

### 1. Code Duplication Issues

**Critical**: ~70% code duplication between:
- `NCYXQuilt` (2D) and `NCZYXQuilt` (3D)
- `LargeNCYXQuilt` (2D) and `LargeNCZYXQuilt` (3D)
- In-memory vs Large (disk-cached) versions

**Specific Duplications:**
- `get_times()` method: Different implementations in 2D vs 3D, but same logic
- `border_tensor()`: Inconsistent return types (torch vs numpy)
- `stitch()` logic: Similar but not identical
- Weight initialization: Repeated in all 4 classes

### 2. Inconsistencies

#### Border Handling
- `NCYXQuilt.border_tensor()`: Returns torch tensor, handles `border=None` case
- `LargeNCYXQuilt.border_tensor()`: Returns numpy array, different logic (`result = result - 1`)
- `NCZYXQuilt.border_tensor()`: Returns torch tensor, doesn't handle `border=None`
- `LargeNCZYXQuilt.border_tensor()`: Returns numpy array, same logic as LargeNCYXQuilt

#### Method Naming
- `unstitch()` vs `unstich_next()` (typo in method name)
- `unstitch_data_pair()` vs `unstitch_and_clean_sparse_data_pair()`

#### Border Parameter Handling
- 2D classes: `border == 0 or border == (0, 0)`
- 3D classes: `border == 0 or border == (0,0,0)` (inconsistent spacing)
- `LargeNCZYXQuilt`: `border=None` default, others require explicit parameter

#### Default Values
- `border_weight`: 0.1 (most) vs 1.0 (`NCYXQuilt`)
- `use_numba`: Only in `NCYXQuilt.stitch()`, not in 3D version

### 3. Architecture Issues

#### Missing Abstraction
- No base class to share common functionality
- Duplicated logic for:
  - Border weight computation
  - Window/step validation
  - Chunk indexing calculations

#### Tight Coupling
- Direct dependencies on implementation details
- Mixed concerns (data handling + disk I/O + computation)

#### Inconsistent Return Types
- Some methods return torch tensors, others numpy arrays
- Mixed use of torch/numpy in same class

### 4. Code Quality Issues

#### Type Hints
- **None present** - Makes code harder to understand and maintain
- No IDE support for autocomplete/type checking

#### Error Handling
- Heavy use of `assert` statements (disabled in optimized Python)
- No proper validation with helpful error messages
- Missing input validation (e.g., window size > 0)

#### Dead/Commented Code
- Line 129-132 in `qlty2DLarge.py`: Commented code
- Line 138 in `qlty3DLarge.py`: Duplicate `unstitch()` call
- Line 242 in `qlty3DLarge.py`: Unused `import os`

#### Documentation
- Inconsistent docstring quality
- Missing parameter descriptions
- Outdated README.md (mentions 3Dlarge, should be 3DLarge)

### 5. Configuration Issues

#### Version Mismatch
- `setup.cfg`: `0.2.0`
- `__init__.py`: `0.3.0`
- `setup.py`: `0.3.0`

#### Empty Requirements
- `setup.py`: `requirements = []` (empty)
- `requirements.txt`: Has actual dependencies
- Not synced

#### Missing Dependencies
- `numba` used but not in `requirements.txt`
- `umap` used in `utils/false_color_2D.py` but not in requirements
- `scikit-learn` used but not explicitly listed

### 6. Testing Gaps

- Only basic tests for 2D/3D in-memory versions
- **No tests** for Large (disk-cached) versions
- No tests for edge cases (empty tensors, invalid parameters)
- No tests for `cleanup.py` functions
- No tests for `utils/false_color_2D.py`

### 7. Performance Issues

#### Inefficient Operations
- `qlty3D.py` stitch: Uses `count` variable instead of direct calculation
- Multiple numpy/torch conversions in `Large` classes
- Redundant `expand_dims` calls in `return_mean()`

#### Memory Issues
- No memory cleanup in Large classes (zarr arrays remain open)
- Iterator reset not handled in `unstich_next()` (typo)

### 8. Module Organization

#### Current Structure
```
qlty/
  ├── __init__.py
  ├── qlty2D.py
  ├── qlty2DLarge.py
  ├── qlty3D.py
  ├── qlty3DLarge.py
  ├── cleanup.py
  └── utils/
      └── false_color_2D.py
```

#### Issues
- Flat structure doesn't reflect relationships
- `utils/` only has one file
- No clear separation of concerns

## Proposed Cleanup and Rework Plan

### Phase 1: Foundation (High Priority)

#### 1.1 Create Base Classes
```python
# qlty/base.py
class BaseQuilt(ABC):
    """Base class for all quilt operations"""
    # Common initialization, validation, weight computation

class BaseQuilt2D(BaseQuilt):
    """Base for 2D operations"""
    # 2D-specific common code

class BaseQuilt3D(BaseQuilt):
    """Base for 3D operations"""
    # 3D-specific common code
```

#### 1.2 Standardize Interfaces
- Consistent method signatures across all classes
- Unified return types (torch tensors for in-memory, numpy for disk)
- Consistent parameter handling

#### 1.3 Fix Critical Bugs
- Fix typo: `unstich_next()` → `unstitch_next()`
- Remove duplicate `unstitch()` call in `qlty3DLarge.py:138`
- Fix incomplete line in `qlty3DLarge.py:58` (weight initialization)
- Handle empty return case in `LargeNCYXQuilt.unstitch_and_clean_sparse_data_pair()`

### Phase 2: Code Consolidation (High Priority)

#### 2.1 Extract Common Logic
- `_compute_chunk_indices()`: Unified chunk calculation
- `_compute_weight_matrix()`: Unified weight computation
- `_validate_parameters()`: Input validation

#### 2.2 Standardize Border Handling
- Single implementation for `border_tensor()`
- Consistent border parameter parsing
- Unified weight matrix generation

#### 2.3 Unify get_times() Implementation
- Use same algorithm in all classes
- Parameterize for 2D vs 3D

### Phase 3: Code Quality (Medium Priority)

#### 3.1 Add Type Hints
```python
from typing import Tuple, Optional, Union
import torch
import numpy as np

def unstitch(
    self,
    tensor: torch.Tensor
) -> torch.Tensor:
    ...
```

#### 3.2 Replace Asserts with Proper Validation
```python
# Instead of:
assert self.border_weight <= 1.0

# Use:
if not (0.0 <= self.border_weight <= 1.0):
    raise ValueError(f"border_weight must be in [0, 1], got {self.border_weight}")
```

#### 3.3 Remove Dead Code
- Remove commented code
- Remove unused imports
- Clean up empty lines

#### 3.4 Improve Documentation
- Add comprehensive docstrings
- Document edge cases
- Add usage examples
- Update README with correct module names

### Phase 4: Configuration & Dependencies (Medium Priority)

#### 4.1 Fix Version Management
- Use single source of truth for version
- Consider using `importlib.metadata` or `setuptools_scm`

#### 4.2 Sync Dependencies
- Move requirements from `requirements.txt` to `setup.py`
- Add missing dependencies (numba, umap-learn, scikit-learn)
- Create `requirements-dev.txt` for development dependencies

#### 4.3 Add py.typed Marker
- For proper type checking support

### Phase 5: Testing (High Priority)

#### 5.1 Expand Test Coverage
- Add tests for Large classes
- Test edge cases (empty tensors, invalid parameters)
- Test cleanup functions
- Test utils module

#### 5.2 Add Integration Tests
- Test full workflow (unstitch → process → stitch)
- Test with real-world data sizes

### Phase 6: Refactoring (Low Priority)

#### 6.1 Reorganize Module Structure
```
qlty/
  ├── __init__.py
  ├── base.py          # Base classes
  ├── core/
  │   ├── __init__.py
  │   ├── quilt_2d.py  # NCYXQuilt
  │   ├── quilt_3d.py  # NCZYXQuilt
  │   └── large/
  │       ├── __init__.py
  │       ├── quilt_2d_large.py
  │       └── quilt_3d_large.py
  ├── utils/
  │   ├── __init__.py
  │   ├── cleanup.py
  │   └── visualization.py  # false_color_2D.py
  └── types.py         # Type aliases and constants
```

#### 6.2 Extract Numba Implementation
- Create separate module for optimized operations
- Make it optional dependency

#### 6.3 Add Context Managers
- For Large classes to ensure proper cleanup of zarr arrays

### Phase 7: Performance Optimization (Low Priority)

#### 7.1 Optimize Stitching
- Vectorize operations where possible
- Reduce numpy/torch conversions
- Optimize weight matrix application

#### 7.2 Memory Management
- Add context managers for resource cleanup
- Implement proper zarr array closing

## Implementation Priority

### Must Fix (Before Next Release)
1. Fix critical bugs (typos, incomplete code)
2. Fix version mismatch
3. Sync dependencies
4. Add tests for Large classes

### Should Fix (Next Release)
1. Create base classes
2. Standardize interfaces
3. Add type hints
4. Remove code duplication

### Nice to Have (Future)
1. Module reorganization
2. Performance optimizations
3. Advanced features

## Migration Strategy

### Backward Compatibility
- Maintain existing class names and signatures
- Add deprecation warnings for old patterns
- Provide migration guide

### Incremental Approach
1. Start with bug fixes (no breaking changes)
2. Add base classes (internal refactoring)
3. Gradually standardize interfaces
4. Add new features

## Estimated Effort

- **Phase 1 (Foundation)**: 2-3 days
- **Phase 2 (Consolidation)**: 3-4 days
- **Phase 3 (Quality)**: 2-3 days
- **Phase 4 (Config)**: 1 day
- **Phase 5 (Testing)**: 3-4 days
- **Phase 6 (Refactoring)**: 4-5 days
- **Phase 7 (Performance)**: 2-3 days

**Total**: ~17-23 days of focused development

## Risk Assessment

### Low Risk
- Bug fixes
- Adding tests
- Documentation improvements

### Medium Risk
- Creating base classes (need to ensure backward compatibility)
- Standardizing interfaces (may break existing code)

### High Risk
- Module reorganization (breaking changes)
- Major API changes

## Success Metrics

1. **Code Reduction**: Reduce codebase size by 30-40% through deduplication
2. **Test Coverage**: Achieve >80% coverage
3. **Type Safety**: Add type hints to all public APIs
4. **Documentation**: All public methods have comprehensive docstrings
5. **Consistency**: All classes follow same patterns and conventions

## Conclusion

The qlty codebase has solid functionality but needs significant cleanup to improve maintainability, consistency, and reliability. The proposed plan addresses code duplication, inconsistencies, and quality issues while maintaining backward compatibility where possible. Starting with bug fixes and foundation work will provide immediate value, while the refactoring phases can be done incrementally.
