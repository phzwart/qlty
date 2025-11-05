#!/usr/bin/env python

"""Tests for Large (disk-cached) quilt classes."""

import os
import shutil
import tempfile

import einops
import numpy as np
import pytest
import torch

from qlty import qlty2DLarge, qlty3DLarge


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.mark.parametrize(
    "step, border",
    [
        ((16, 32), None),
        ((16, 32), (0, 0)),
        ((8, 8), (2, 3)),
    ],
)
def test_LargeNCYXQuilt(temp_dir, step, border):
    """Test LargeNCYXQuilt basic functionality."""
    x = np.linspace(0, np.pi * 2.0, 128)
    X, Y = np.meshgrid(x, x)
    imgs = []

    for ii in range(5):
        img = []
        for jj in range(3):
            tmp = np.sin((jj + 1) * X + ii * np.pi / 3.0) + np.cos(
                (ii + 1) * Y + np.pi * jj / 3.0
            )
            img.append(tmp)
        img = torch.Tensor(einops.rearrange(img, "C Y X -> C Y X"))
        imgs.append(img)

    imgs_in = einops.rearrange(imgs, "N C Y X -> N C Y X")
    _ = einops.reduce(imgs_in, "N C Y X -> N () Y X", reduction="sum")  # noqa: F841

    filename = os.path.join(temp_dir, "test_2d")
    quilt = qlty2DLarge.LargeNCYXQuilt(
        filename=filename,
        N=5,
        Y=128,
        X=128,
        window=(16, 32),
        step=step,
        border=border,
        border_weight=0.07,
    )

    # Test unstitch_and_clean_sparse_data_pair
    missing_label = -1
    labels = torch.zeros((5, 128, 128)) + missing_label
    labels[:, 10:118, 10:118] = 1.0  # Some valid data
    labels = labels.unsqueeze(1)

    ain, aout = quilt.unstitch_and_clean_sparse_data_pair(
        imgs_in, labels, missing_label
    )

    # Verify we got some patches
    assert len(ain) > 0 or isinstance(ain, torch.Tensor)
    if isinstance(ain, torch.Tensor):
        assert ain.shape[0] > 0

    # Test stitching process
    for ii in range(quilt.N_chunks):
        ind, tmp = quilt.unstitch_next(imgs_in)
        # Simulate neural network output
        neural_network_result = tmp.unsqueeze(0)
        quilt.stitch(neural_network_result, ii)

    # Get mean result
    mean = quilt.return_mean()
    assert mean.shape == (5, 3, 128, 128)

    # Clean up
    for suffix in [
        "_mean_cache.zarr",
        "_std_cache.zarr",
        "_norma_cache.zarr",
        "_mean.zarr",
    ]:
        path = filename + suffix
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize(
    "step, border",
    [
        ((16, 16, 16), None),
        ((16, 16, 16), (0, 0, 0)),
        ((7, 7, 7), (1, 1, 3)),
    ],
)
def test_LargeNCZYXQuilt(temp_dir, step, border):
    """Test LargeNCZYXQuilt basic functionality."""
    x = np.linspace(0, np.pi * 2.0, 64)  # Smaller for 3D
    X, Y, Z = np.meshgrid(x, x, x)
    imgs = []

    for ii in range(2):
        img = []
        for jj in range(2):
            tmp = (
                np.sin((jj + 1) * X + ii * np.pi / 3.0)
                + np.cos((ii + 1) * Y + np.pi * jj / 3.0)
                + np.cos((ii - jj) * Z + (ii + jj) * np.pi / 5.0)
            )
            img.append(tmp)
        img = torch.Tensor(einops.rearrange(img, "C Z Y X -> C Z Y X"))
        imgs.append(img)

    imgs_in = einops.rearrange(imgs, "N C Z Y X -> N C Z Y X")

    filename = os.path.join(temp_dir, "test_3d")
    quilt = qlty3DLarge.LargeNCZYXQuilt(
        filename=filename,
        N=2,
        Z=64,
        Y=64,
        X=64,
        window=(16, 16, 16),
        step=step,
        border=border,
        border_weight=0.07,
    )

    # Test unstitch_and_clean_sparse_data_pair
    missing_label = -1
    labels = torch.zeros((2, 64, 64, 64)) + missing_label
    labels[:, 5:59, 5:59, 5:59] = 1.0  # Some valid data
    labels = labels.unsqueeze(1)

    ain, aout = quilt.unstitch_and_clean_sparse_data_pair(
        imgs_in, labels, missing_label
    )

    # Verify we got some patches
    assert len(ain) > 0 or isinstance(ain, torch.Tensor)
    if isinstance(ain, torch.Tensor):
        assert ain.shape[0] > 0

    # Test stitching process
    for ii in range(quilt.N_chunks):
        ind, tmp = quilt.unstitch_next(imgs_in)
        neural_network_result = tmp.unsqueeze(0)
        quilt.stitch(neural_network_result, ii)

    # Get mean result
    mean = quilt.return_mean()
    assert mean.shape == (2, 2, 64, 64, 64)

    # Clean up
    for suffix in [
        "_mean_cache.zarr",
        "_std_cache.zarr",
        "_norma_cache.zarr",
        "_mean.zarr",
    ]:
        path = filename + suffix
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


def test_LargeNCYXQuilt_empty_patches(temp_dir):
    """Test LargeNCYXQuilt with all patches being empty (missing label)."""
    filename = os.path.join(temp_dir, "test_empty")
    quilt = qlty2DLarge.LargeNCYXQuilt(
        filename=filename,
        N=2,
        Y=100,
        X=100,
        window=(50, 50),
        step=(25, 25),
        border=(10, 10),
        border_weight=0.1,
    )

    data = torch.randn(2, 3, 100, 100)
    labels = torch.zeros((2, 100, 100)) - 1  # All missing
    labels = labels.unsqueeze(1)

    ain, aout = quilt.unstitch_and_clean_sparse_data_pair(data, labels, -1)

    # Should return empty lists when no valid patches
    assert len(ain) == 0 or (isinstance(ain, list) and len(ain) == 0)

    # Clean up
    for suffix in [
        "_mean_cache.zarr",
        "_std_cache.zarr",
        "_norma_cache.zarr",
        "_mean.zarr",
    ]:
        path = filename + suffix
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


def test_LargeNCYXQuilt_with_std(temp_dir):
    """Test LargeNCYXQuilt return_mean with std=True."""
    filename = os.path.join(temp_dir, "test_std")
    quilt = qlty2DLarge.LargeNCYXQuilt(
        filename=filename,
        N=2,
        Y=64,
        X=64,
        window=(32, 32),
        step=(16, 16),
        border=(5, 5),
        border_weight=0.1,
    )

    data = torch.randn(2, 1, 64, 64)

    # Process all chunks
    for ii in range(quilt.N_chunks):
        ind, tmp = quilt.unstitch_next(data)
        result = tmp.unsqueeze(0)
        # Provide variance
        var = torch.ones_like(result) * 0.1
        quilt.stitch(result, ii, patch_var=var)

    mean, std = quilt.return_mean(std=True)
    assert mean.shape == (2, 1, 64, 64)
    assert std.shape == (2, 1, 64, 64)

    # Clean up
    for suffix in [
        "_mean_cache.zarr",
        "_std_cache.zarr",
        "_norma_cache.zarr",
        "_mean.zarr",
        "_std.zarr",
    ]:
        path = filename + suffix
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
