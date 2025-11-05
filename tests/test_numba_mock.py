#!/usr/bin/env python

"""Tests for numba availability and usage.

Note: Numba is imported at the top level in qlty2D.py, so we cannot easily
mock it being unavailable without significant refactoring. However, we can
test that both use_numba=True and use_numba=False paths work correctly.

The numba_njit_stitch function (lines 20-34) is JIT-compiled and cannot be
directly tested, but it is exercised through the use_numba=True path.
"""

import pytest
import torch

from qlty import qlty2D


def test_stitch_with_numba_available():
    """Test that code works when numba is available (normal case)."""
    try:
        import numba  # noqa: F401
    except ImportError:
        pytest.skip("Numba is not available in this environment")

    quilt = qlty2D.NCYXQuilt(
        Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5), border_weight=0.1
    )

    data = torch.randn(2, 3, 128, 128)
    patches = quilt.unstitch(data)

    # Should work with numba (default)
    # This exercises the numba_njit_stitch function indirectly
    reconstructed, weights = quilt.stitch(patches, use_numba=True)
    assert reconstructed.shape == data.shape
    assert weights.shape == (128, 128)
