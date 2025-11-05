# QLTY Cleanup Action Plan

## Immediate Fixes (Do First)

### 1. Critical Bugs

#### Bug #1: Typo in method name
- **File**: `qlty2DLarge.py:200`, `qlty3DLarge.py:205`
- **Issue**: Method named `unstich_next()` should be `unstitch_next()`
- **Fix**: Rename method and all call sites
- **Impact**: Breaking change, but clearly a bug

#### Bug #2: Duplicate unstitch call
- **File**: `qlty3DLarge.py:138`
- **Issue**: `tmp = self.unstitch(tensor_in, ii)` is assigned but never used, then `unstitched_in.append(self.unstitch(tensor_in, ii))` is called again
- **Fix**: Remove line 138, keep only the append line
- **Impact**: Performance improvement, no functional change

#### Bug #3: Incomplete weight initialization (if exists)
- **File**: `qlty3DLarge.py:58` (check if line exists)
- **Issue**: Need to verify weight initialization is complete
- **Fix**: Ensure weight is properly initialized

### 2. Configuration Issues

#### Config #1: Version mismatch
- **Files**: `setup.cfg:2` (0.2.0), `__init__.py:5` (0.3.0), `setup.py:42` (0.3.0)
- **Fix**: Update `setup.cfg` to 0.3.0 or use single source of truth

#### Config #2: Empty requirements in setup.py
- **File**: `setup.py:13`
- **Issue**: `requirements = []` but dependencies exist in `requirements.txt`
- **Fix**: Read from `requirements.txt` or hardcode dependencies

#### Config #3: Missing dependencies
- **Files**: `requirements.txt`
- **Issue**: Missing `numba`, `umap-learn`, `scikit-learn`
- **Fix**: Add all dependencies used in code

### 3. Code Quality Quick Wins

#### Quality #1: Remove unused import
- **File**: `qlty3DLarge.py:242`
- **Issue**: `import os` is imported but never used
- **Fix**: Remove the import

#### Quality #2: Remove commented code
- **File**: `qlty2DLarge.py:129-132`
- **Issue**: Commented out code blocks
- **Fix**: Remove commented code

#### Quality #3: Fix inconsistent border_tensor return types
- **Files**: All quilt classes
- **Issue**: Some return torch.Tensor, others return np.ndarray
- **Fix**: Standardize - in-memory classes use torch, Large classes use numpy

## Short-term Improvements (Next Sprint)

### 1. Standardize Method Signatures

#### Unify border parameter handling
```python
# Current: Mixed approaches
if border == 0 or border == (0, 0):
if border == 0:
if border == 0 or border == (0,0,0):

# Proposed: Helper function
def _normalize_border(border, ndim):
    if border is None or border == 0:
        return None
    if isinstance(border, int):
        return tuple([border] * ndim)
    if isinstance(border, tuple) and all(b == 0 for b in border):
        return None
    return border
```

#### Standardize default border_weight
- Current: 0.1 (most), 1.0 (NCYXQuilt)
- Proposed: Use 0.1 everywhere (NCYXQuilt default seems wrong)

### 2. Create Shared Utilities

#### Extract common validation
```python
# qlty/utils/validation.py
def validate_quilt_params(window, step, border, border_weight, ndim):
    """Validate common quilt parameters"""
    # Check window size
    # Check step size
    # Check border
    # Check border_weight range
    pass
```

#### Extract weight computation
```python
# qlty/utils/weights.py
def compute_weight_matrix(window, border, border_weight, use_torch=True):
    """Compute weight matrix for stitching"""
    # Unified implementation
    pass
```

### 3. Add Type Hints (Start with Public API)

```python
from typing import Tuple, Optional, Union
import torch
import numpy as np

def unstitch(
    self, 
    tensor: torch.Tensor
) -> torch.Tensor:
    """Unstitch a tensor into patches."""
    ...
```

## Medium-term Refactoring (Next Release)

### 1. Create Base Classes

See `CODEBASE_REVIEW.md` Phase 1 for detailed plan.

### 2. Consolidate Duplicate Code

- Extract `get_times()` logic
- Unify `border_tensor()` implementation
- Standardize `stitch()` methods

### 3. Expand Testing

- Add tests for Large classes
- Test edge cases
- Test cleanup functions

## Code Review Checklist

When reviewing/merging changes, check:

- [ ] No code duplication (use shared utilities)
- [ ] Consistent return types (torch vs numpy)
- [ ] Type hints on public methods
- [ ] Proper error handling (no bare asserts)
- [ ] Tests added for new/changed code
- [ ] Documentation updated
- [ ] Dependencies updated if needed
- [ ] Version bumped if needed

## Quick Reference: File-by-File Issues

### qlty2D.py
- ✅ `get_times()`: Good implementation with edge case handling
- ✅ `border_tensor()`: Handles None case properly
- ⚠️ `border_weight` default is 1.0 (inconsistent)
- ⚠️ Missing type hints

### qlty2DLarge.py
- ❌ Typo: `unstich_next()` → `unstitch_next()`
- ❌ Commented code at lines 129-132
- ⚠️ `border_tensor()` returns numpy (should be consistent with naming)
- ⚠️ Missing type hints
- ⚠️ No error handling for empty unstitch results

### qlty3D.py
- ⚠️ `get_times()`: Simpler implementation (no edge case handling like 2D)
- ⚠️ `border_tensor()`: Doesn't handle None case
- ⚠️ `stitch()`: Uses `count` variable (inefficient)
- ⚠️ Missing type hints

### qlty3DLarge.py
- ❌ Typo: `unstich_next()` → `unstitch_next()`
- ❌ Duplicate `unstitch()` call at line 138
- ❌ Unused import: `os` at line 242
- ⚠️ Missing type hints

### cleanup.py
- ⚠️ Missing type hints
- ⚠️ No tests
- ⚠️ Similar functions could be unified

### utils/false_color_2D.py
- ⚠️ Missing type hints
- ⚠️ No tests
- ⚠️ Missing dependencies in requirements.txt

## Testing Strategy

### Phase 1: Fix Existing Tests
- Ensure all current tests pass
- Fix any broken tests

### Phase 2: Add Missing Tests
```python
# tests/test_qlty2DLarge.py
def test_large_2d_basic():
    """Test basic LargeNCYXQuilt functionality"""
    ...

def test_large_2d_edge_cases():
    """Test edge cases"""
    ...

# tests/test_qlty3DLarge.py
# Similar structure

# tests/test_cleanup.py
def test_weed_sparse_2d():
    """Test 2D sparse data weeding"""
    ...
```

### Phase 3: Integration Tests
- Test full workflow
- Test with realistic data sizes
- Test error conditions

## Documentation Updates Needed

1. **README.md**
   - Fix typo: "unstich" → "unstitch"
   - Update module names (3Dlarge → 3DLarge)
   - Add missing dependencies

2. **Docstrings**
   - Add parameter types
   - Add return types
   - Add examples
   - Document edge cases

3. **API Documentation**
   - Generate from docstrings
   - Add usage examples
   - Document migration path

## Migration Notes

### For Users
- No breaking changes in immediate fixes
- Method rename (`unstich_next` → `unstitch_next`) is a bug fix
- Future refactoring will maintain backward compatibility

### For Developers
- Use base classes when adding new functionality
- Follow type hint guidelines
- Add tests for all new code
- Use shared utilities instead of duplicating code

