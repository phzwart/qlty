# Coverage Testing

## Running Coverage

Due to a conflict between `pytest-cov` and torch imports in this environment, we use `coverage run` directly instead of `pytest --cov`:

```bash
# Run tests with coverage
python -m coverage run --source=qlty -m pytest tests/ -v

# Generate coverage report
python -m coverage report --show-missing

# Generate HTML report
python -m coverage html
```

## Coverage Results

Current coverage for target modules:
- `qlty2_5D.py`: 88% (improved from 75%)
- `backends_2_5D.py`: 70% (improved from 62%)
- `stack_to_zarr.py`: 94% (improved from 38%)

## Why not pytest-cov?

When `pytest-cov` instruments modules, it can trigger torch imports in a way that causes the error:
```
RuntimeError: function '_has_torch_function' already has a docstring
```

Using `coverage run` directly avoids this issue by collecting coverage data without the module instrumentation conflicts.
