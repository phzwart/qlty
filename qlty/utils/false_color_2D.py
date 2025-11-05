import einops
import numpy as np
import torch
import umap
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

from qlty.qlty2D import NCYXQuilt


class FalseColorGenerator:
    def __init__(
        self, image_shape, window_size=32, step_size=8, reducer=None, scaler=None
    ):
        self.image_shape = image_shape
        self.qlty_object = NCYXQuilt(
            X=image_shape[-1],
            Y=image_shape[-2],
            window=(window_size, window_size),
            step=(step_size, step_size),
            border=0,
        )
        # Precompute patch coordinates
        tmp_x = np.arange(0, image_shape[-1], 1)
        tmp_y = np.arange(0, image_shape[-2], 1)
        self.Y, self.X = np.meshgrid(tmp_y, tmp_x, indexing="ij")

        self.Y = torch.Tensor(self.Y).unsqueeze(0).unsqueeze(0)
        self.X = torch.Tensor(self.X).unsqueeze(0).unsqueeze(0)

        self.patch_X = self.qlty_object.unstitch(self.X)
        self.patch_Y = self.qlty_object.unstitch(self.Y)
        self.mean_patch_X = torch.mean(self.patch_X, dim=(-1, -2))
        self.mean_patch_Y = torch.mean(self.patch_Y, dim=(-1, -2))

        # for NN interpolation
        YX = einops.rearrange(
            torch.cat([self.Y, self.X], dim=1), "N C Y X -> (N Y X) C"
        )
        pYX = torch.cat([self.mean_patch_Y, self.mean_patch_X], dim=1).numpy()
        YX = YX.numpy()
        tree = cKDTree(pYX)
        dist, self.idx = tree.query(YX, k=1)

        self.reducer = reducer
        self.scaler = scaler
        self.reducer_is_trained = True
        self.scaler_is_trained = True

        if self.reducer is None:
            self.reducer = umap.UMAP(n_components=3)
            self.reducer_is_trained = False

        if self.scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler_is_trained = False

    def train_reducer_from_patches(self, selected_patches):
        print(selected_patches.shape)
        lin_patches = einops.rearrange(selected_patches, "N C Y X -> N (C Y X)")
        tmp = self.reducer.fit_transform(lin_patches)
        self.reducer_is_trained = True
        tmp = self.scaler.fit_transform(tmp)

    def train_reducer(self, images, num_patches=None):
        assert len(images.shape) == 4
        patches = self.qlty_object.unstitch(images)
        N_patches = patches.shape[0]
        if num_patches is None:
            num_patches = N_patches
        sel = np.argsort(np.random.uniform(0, 1, N_patches))[:num_patches]
        patches = patches[sel]
        self.train_reducer_from_patches(patches)
        self.scaler_is_trained = True

    def __call__(self, image):
        assert image.shape[0] == 1
        patches = self.qlty_object.unstitch(image)
        lin_patches = einops.rearrange(patches, "N C Y X -> N (C Y X)")
        UVW = self.reducer.transform(lin_patches)
        if self.scaler_is_trained:
            UVW = self.scaler.transform(UVW)
        else:
            UVW = self.scaler.fit_transform(UVW)
            self.scaler_is_trained = True

        interpolated_RGB = UVW[self.idx]

        H, W = self.image_shape[-2], self.image_shape[-1]
        interpolated_RGB = einops.rearrange(
            interpolated_RGB, "(Y X) C -> Y X C", X=W, Y=H
        )
        sel = interpolated_RGB > 1
        interpolated_RGB[sel] = 1

        return interpolated_RGB
