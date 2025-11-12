import numpy as np
import pytest

from qlty import qlty2D

pytest.importorskip("einops")
torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "window,step",
    [
        ((16, 32), (8, 8)),  # parity coloring sufficient
        ((16, 32), (4, 8)),  # requires four colors along Y
    ],
)
def test_stitch_parallel_colored_matches_serial_dense_overlap(window, step):
    torch.manual_seed(0)

    Y, X = 80, 96
    n_images = 2
    channels = 3

    quilt = qlty2D.NCYXQuilt(Y=Y, X=X, window=window, step=step, border=(2, 2))

    data = torch.randn(n_images, channels, Y, X)
    patches = quilt.unstitch(data)

    serial_result, serial_norma = qlty2D.stitch_serial_numba(
        patches.clone(),
        quilt.weight,
        quilt.window,
        quilt.step,
        quilt.Y,
        quilt.X,
        quilt.nY,
        quilt.nX,
    )

    parallel_result, parallel_norma = qlty2D.stitch_parallel_colored(
        patches.clone(),
        quilt.weight,
        quilt.window,
        quilt.step,
        quilt.Y,
        quilt.X,
        quilt.nY,
        quilt.nX,
    )

    assert torch.allclose(parallel_result, serial_result, atol=1e-6, rtol=1e-5)
    assert torch.allclose(parallel_norma, serial_norma, atol=1e-6, rtol=1e-5)


def test_ensure_numpy_passthrough():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    returned = qlty2D._ensure_numpy(arr)
    assert returned is arr
