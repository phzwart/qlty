"""
False color generation utility for 2D images.

Generates false-color visualizations of 2D images using UMAP dimensionality
reduction to map image patches to RGB color space for visualization purposes.
"""

import einops
import numpy as np
import torch
import umap
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

from qlty.qlty2D import NCYXQuilt


class FalseColorGenerator:
    """
    Generate false-color visualizations of 2D images using UMAP.

    This class uses patch-based dimensionality reduction to convert image patches
    into RGB color space for visualization. Patches are extracted using a sliding
    window approach, reduced to 3D using UMAP, and then interpolated back to
    full image resolution for visualization.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        Shape of input images, typically (1, C, Y, X) or (N, C, Y, X).
        Last two dimensions (Y, X) are used for spatial dimensions.
    window_size : int, optional
        Size of sliding window for patch extraction, by default 32
    step_size : int, optional
        Step size for sliding window, by default 8
    reducer : umap.UMAP, optional
        Pre-initialized UMAP reducer. If None, creates new one with n_components=3.
    scaler : sklearn.preprocessing.MinMaxScaler, optional
        Pre-initialized MinMaxScaler. If None, creates new one.

    Examples
    --------
    >>> generator = FalseColorGenerator(image_shape=(1, 1, 256, 256))
    >>> training_images = torch.randn(4, 1, 256, 256)
    >>> generator.train_reducer(training_images)
    >>> test_image = torch.randn(1, 1, 256, 256)
    >>> rgb_image = generator(test_image)  # Shape: (256, 256, 3)
    """

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
        """
        Train the reducer and scaler from a set of selected patches.

        Parameters
        ----------
        selected_patches : torch.Tensor
            Patches to train on, shape (N, C, Y, X) where:
            - N: number of patches
            - C: number of channels
            - Y, X: spatial dimensions of each patch

        Notes
        -----
        This method fits both the reducer (UMAP) and scaler (MinMaxScaler)
        on the provided patches. Patches are flattened before reduction.
        """
        print(selected_patches.shape)
        lin_patches = einops.rearrange(selected_patches, "N C Y X -> N (C Y X)")
        tmp = self.reducer.fit_transform(lin_patches)
        self.reducer_is_trained = True
        tmp = self.scaler.fit_transform(tmp)

    def train_reducer(self, images, num_patches=None):
        """
        Train the reducer from a set of images by extracting patches.

        Parameters
        ----------
        images : torch.Tensor
            Training images, shape (N, C, Y, X) where:
            - N: batch size
            - C: number of channels
            - Y, X: spatial dimensions
        num_patches : int, optional
            Number of patches to use for training. If None, uses all patches.
            By default None

        Notes
        -----
        Patches are randomly selected if num_patches is less than the total
        number of patches available. This is useful for large images where
        using all patches would be computationally expensive.
        """
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
        """
        Generate false-color visualization of an input image.

        Parameters
        ----------
        image : torch.Tensor
            Input image to visualize, shape (1, C, Y, X).
            Batch size must be 1.

        Returns
        -------
        numpy.ndarray
            False-color RGB image, shape (Y, X, 3). Values are in [0, 1].

        Raises
        ------
        AssertionError
            If image batch size is not 1, or if reducer is not trained.

        Notes
        -----
        The process involves:
        1. Extracting patches from the input image
        2. Reducing patches to 3D using trained UMAP
        3. Scaling to [0, 1] range
        4. Interpolating back to full image resolution using nearest neighbors
        5. Clipping any values > 1 to ensure valid RGB range

        The scaler will be auto-trained if not already trained, but the reducer
        must be trained beforehand using train_reducer() or train_reducer_from_patches().
        """
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
