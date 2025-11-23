# Coverage Reporting Notes

## Current Coverage (Verified)

When running the full test suite locally, coverage is:

- `qlty/utils/stack_to_zarr.py`: **94.30%** (158 statements, 9 missing)
- `qlty/qlty2_5D.py`: **88.26%** (443 statements, 52 missing)
- `qlty/backends_2_5D.py`: **70.46%** (237 statements, 70 missing)

## Codecov Diff View

Codecov shows coverage in a **diff view** comparing your PR/branch against a baseline commit.
If Codecov shows lower coverage numbers, it may be because:

1. **Old Baseline**: The baseline commit (`066eddb` in your case) had the old importlib code
   that bypassed coverage tracking, showing only ~38% coverage
2. **New Coverage Not Processed Yet**: Codecov needs to process the new coverage.xml from CI
3. **Comparison Mode**: Codecov shows "Patch" coverage (changes only) vs "Overall" coverage

## Verification

To verify coverage locally:

```bash
# Run all tests with coverage
python -m coverage run --source=qlty -m pytest tests/ -v

# Generate report
python -m coverage report --show-missing

# Generate XML (what Codecov uses)
python -m coverage xml

# Check specific modules
python -m coverage report --include="qlty/utils/stack_to_zarr.py" --show-missing
```

The coverage.xml file generated contains the correct coverage data. Codecov will update
its baseline once it processes the new data from the CI run.

