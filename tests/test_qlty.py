#!/usr/bin/env python

"""Tests for `qlty` package."""

import numpy as np
import torch
import einops
from qlty import qlty2D
from qlty import qlty3D


def test_NCYXQuilt():
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
    quilt = qlty2D.NCYXQuilt(Y=128, X=128, window=(16, 32), step=(8, 16), border=(2, 4), border_weight=0.07)
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
    return True


def test_NCZYXQuilt():
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
                              step=(8, 8, 8),
                              border=(8, 8, 8),
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

    return True

def run_tests():
    test_NCYXQuilt()
    test_NCZYXQuilt()


if __name__ =="__main__":
    run_tests()
    print("OK")
