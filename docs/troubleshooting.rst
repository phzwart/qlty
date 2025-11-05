Troubleshooting
===============

Common Issues and Solutions
----------------------------

Issue: AssertionError when stitching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Getting `AssertionError` when calling `stitch()`.

**Cause**: The number of patches doesn't match the expected number based on the quilt configuration.

**Solution**: Make sure the number of patches is exactly `N * nY * nX` (for 2D) or `N * nZ * nY * nX` (for 3D), where N is the number of images::

    quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5))
    nY, nX = quilt.get_times()
    expected_patches = num_images * nY * nX

    patches = quilt.unstitch(data)
    assert patches.shape[0] == expected_patches, \
        f"Expected {expected_patches} patches, got {patches.shape[0]}"

Issue: Memory errors with large datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Running out of memory when processing large datasets.

**Solution**: Use the Large classes with disk caching::

    # Instead of:
    quilt = NCYXQuilt(...)  # Loads everything into memory

    # Use:
    quilt = LargeNCYXQuilt(filename, N, Y, X, ...)  # Uses disk caching

Issue: Border artifacts in reconstructed images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Visible artifacts or seams at patch boundaries.

**Solutions**:
1. Increase overlap (reduce step size)::

       quilt = NCYXQuilt(..., step=(window[0]//2, window[1]//2), ...)

2. Increase border size::

       quilt = NCYXQuilt(..., border=(10, 10), ...)  # Larger border

3. Decrease border weight::

       quilt = NCYXQuilt(..., border_weight=0.05, ...)  # More downweighting

Issue: Softmax giving wrong results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Probabilities don't sum to 1 or seem incorrect after stitching.

**Cause**: Applying softmax before stitching instead of after.

**Solution**: Always stitch logits, then apply softmax::

    # WRONG:
    probs = F.softmax(model(patches), dim=1)
    result, _ = quilt.stitch(probs)  # Averaging probabilities is wrong!

    # CORRECT:
    logits = model(patches)
    stitched_logits, _ = quilt.stitch(logits)
    result = F.softmax(stitched_logits, dim=1)  # Apply softmax after

Issue: Numba compilation errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Getting Numba compilation warnings or errors.

**Solution**: Disable Numba and use pure PyTorch::

    result, weights = quilt.stitch(patches, use_numba=False)

Issue: Zarr cache files not cleaned up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Large Zarr cache files remaining after processing.

**Solution**: Manually clean up cache files::

    import os
    import shutil

    for suffix in ["_mean_cache.zarr", "_std_cache.zarr", "_norma_cache.zarr",
                   "_mean.zarr", "_std.zarr"]:
        path = filename + suffix
        if os.path.exists(path):
            shutil.rmtree(path)

Issue: Empty patches after filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: All patches filtered out when using `weed_sparse_classification_training_pairs_2D`.

**Cause**: No valid data in any patches (all missing labels or only in border regions).

**Solution**: Check your data and border settings::

    # Check if any patches have valid data
    valid_count = (labels != missing_label).sum()
    if valid_count == 0:
        print("Warning: No valid data found!")

    # Check border tensor
    border_tensor = quilt.border_tensor()
    valid_pixels = border_tensor.sum()
    print(f"Valid (non-border) pixels per patch: {valid_pixels}")

Issue: Wrong tensor shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Getting shape mismatch errors.

**Common causes**:
1. Wrong input shape - expect `(N, C, Y, X)` for 2D or `(N, C, Z, Y, X)` for 3D
2. Wrong number of dimensions
3. Mismatch between quilt configuration and actual data size

**Solution**: Verify shapes match::

    # 2D: Should be (N, C, Y, X)
    assert len(data.shape) == 4, f"Expected 4D tensor, got {len(data.shape)}D"
    assert data.shape[2] == quilt.Y, f"Y dimension mismatch"
    assert data.shape[3] == quilt.X, f"X dimension mismatch"

    # 3D: Should be (N, C, Z, Y, X)
    assert len(data.shape) == 5, f"Expected 5D tensor, got {len(data.shape)}D"

Issue: Slow stitching performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Stitching is slow for large numbers of patches.

**Solutions**:
1. Enable Numba (default for 2D)::

       result, weights = quilt.stitch(patches, use_numba=True)

2. Use batch processing
3. Consider using Large classes for very large datasets

Issue: Border weight not working as expected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Border regions still have too much influence.

**Solution**: Check that border is set correctly::

    # Verify border is set
    assert quilt.border is not None, "Border is None!"

    # Check weight matrix
    weight = quilt.weight
    print(f"Border weight: {weight[0, 0].item()}")  # Should be border_weight
    print(f"Center weight: {weight[quilt.border[0], quilt.border[1]].item()}")  # Should be 1.0

Issue: get_times() returns unexpected values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Number of patches doesn't match expectations.

**Cause**: The algorithm ensures the last patch fits, which may require adjusting the starting position.

**Solution**: The algorithm is correct - it ensures full coverage. Verify::

    nY, nX = quilt.get_times()
    # Check that we can cover the full image
    last_y_start = (nY - 1) * quilt.step[0]
    last_x_start = (nX - 1) * quilt.step[1]
    assert last_y_start + quilt.window[0] <= quilt.Y
    assert last_x_start + quilt.window[1] <= quilt.X

Performance Tips
----------------

1. **Use appropriate batch sizes**: Process patches in batches for better GPU utilization

2. **Enable Numba**: For 2D stitching, Numba provides significant speedup

3. **Choose step size wisely**:
   - Smaller step = more overlap = smoother results but slower
   - Larger step = less overlap = faster but may have artifacts

4. **Memory management**:
   - Use in-memory classes for small datasets
   - Use Large classes for datasets > several GB

5. **Zarr chunking**: Large classes automatically optimize Zarr chunk sizes

Getting Help
------------

If you encounter issues not covered here:

1. Check the examples in `docs/examples.rst`
2. Review the API documentation in `docs/api.rst`
3. Check existing issues on GitHub
4. Open a new issue with:
   - Minimal reproducible example
   - Error messages
   - Your environment (Python version, PyTorch version, etc.)
