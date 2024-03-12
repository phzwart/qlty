#!/usr/bin/env python

"""Tests for `qlty` package."""
import pytest
import numpy as np
import torch
import einops
from qlty import qlty2D
from qlty import qlty3D


@pytest.mark.parametrize("step, border", [((16, 32),(None)),
                                          ((16,32),(0,0)),
                                          ((8,8),(2,3))])
def test_NCYXQuilt(step, border):
    x = np.linspace(0, np.pi * 2.0, 128)
    X, Y = np.meshgrid(x, x)
    imgs = []

    for ii in range(10):
        img = []
        for jj in range(3):
            tmp = np.sin((jj + 1) * X + ii * np.pi / 3.0) + np.cos((ii + 1) * Y + np.pi * jj / 3.0)
            img.append(tmp)
        img = torch.Tensor(einops.rearrange(img, "C Y X -> C Y X"))
        imgs.append(img)

    imgs_in = einops.rearrange(imgs, "N C Y X -> N C Y X")
    imgs_out = einops.reduce(imgs_in, "N C Y X -> N () Y X", reduction='sum')
    quilt = qlty2D.NCYXQuilt(Y=128, X=128, window=(16, 32), step=step, border=border, border_weight=0.07)
    ain, aout = quilt.unstitch_data_pair(imgs_in, imgs_out)

    reconstruct_in, win = quilt.stitch(ain)
    reconstruct_out, wout = quilt.stitch(aout)

    for ii in range(10):
        reco_in = reconstruct_in[ii, ...]
        orig_in = imgs_in[ii, ...]
        reco_out = reconstruct_out[ii, ...]
        orig_out = imgs_out[ii, ...]

        delta_in = torch.mean(torch.abs(reco_in - orig_in)).item() / ain.shape[0]
        assert (delta_in < 1e-7)
        delta_out = torch.mean(torch.abs(reco_out - orig_out)).item() / aout.shape[0]
        assert (delta_out < 1e-7)



@pytest.mark.parametrize("step, border", [((16, 16, 16),(None)),
                                          ((16, 16, 16),(0,0,0)),
                                          ((7,7,7),(1,1,3))])
def test_NCZYXQuilt(step, border):
    x = np.linspace(0, np.pi * 2.0, 128)
    X, Y, Z = np.meshgrid(x, x, x)
    imgs = []

    for ii in range(3):
        img = []
        for jj in range(3):
            tmp = np.sin((jj + 1) * X + ii * np.pi / 3.0) + np.cos((ii + 1) * Y + np.pi * jj / 3.0) + np.cos(
                (ii - jj) * Z + (ii + jj) * np.pi / 5.0)
            img.append(tmp)
        img = torch.Tensor(einops.rearrange(img, "C Z Y X -> C Z Y X"))
        imgs.append(img)

    imgs_in = einops.rearrange(imgs, "N C Z Y X -> N C Z Y X")
    imgs_out = einops.reduce(imgs_in, "N C Z Y X -> N () Z Y X", reduction='sum')
    quilt = qlty3D.NCZYXQuilt(Z=128,
                              Y=128,
                              X=128,
                              window=(16, 16, 16),
                              step=step,
                              border=border,
                              border_weight=0.07)

    ain, aout = quilt.unstitch_data_pair(imgs_in, imgs_out)
    reconstruct_in, win = quilt.stitch(ain)
    reconstruct_out, wout = quilt.stitch(aout)

    for ii in range(3):
        reco_in = reconstruct_in[ii, ...]
        orig_in = imgs_in[ii, ...]
        reco_out = reconstruct_out[ii, ...]
        orig_out = imgs_out[ii, ...]

        delta_in = torch.mean(torch.abs(reco_in - orig_in)).item() / ain.shape[0]
        assert (delta_in < 1e-7)
        delta_out = torch.mean(torch.abs(reco_out - orig_out)).item() / aout.shape[0]
        assert (delta_out < 1e-7)
