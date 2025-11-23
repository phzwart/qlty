"""
Tests for false_color_2D utility module.
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

try:
    import umap
except ImportError:
    umap = None

try:
    from qlty.utils.false_color_2D import FalseColorGenerator
except ImportError:
    pytest.skip("false_color_2D module not available", allow_module_level=True)


@pytest.fixture
def sample_image_shape():
    """Return a sample image shape (N=1, C=1, Y=64, X=64)."""
    return (1, 1, 64, 64)


@pytest.fixture
def sample_image(sample_image_shape):
    """Create a sample 2D image tensor."""
    return torch.randn(*sample_image_shape)


@pytest.fixture
def sample_images():
    """Create multiple sample images for training."""
    return torch.randn(4, 1, 64, 64)


def test_false_color_generator_init_default(sample_image_shape):
    """Test FalseColorGenerator initialization with default parameters."""
    generator = FalseColorGenerator(sample_image_shape)

    assert generator.image_shape == sample_image_shape
    assert generator.qlty_object.window == (32, 32)  # Default window_size
    assert generator.qlty_object.step == (8, 8)  # Default step_size
    assert generator.reducer is not None
    assert generator.scaler is not None
    assert not generator.reducer_is_trained
    assert not generator.scaler_is_trained
    assert generator.qlty_object is not None


def test_false_color_generator_init_custom_window_step(sample_image_shape):
    """Test initialization with custom window and step sizes."""
    generator = FalseColorGenerator(sample_image_shape, window_size=16, step_size=4)

    assert generator.qlty_object.window == (16, 16)
    assert generator.qlty_object.step == (4, 4)


def test_false_color_generator_init_with_reducer_scaler(sample_image_shape):
    """Test initialization with pre-provided reducer and scaler."""
    if umap is None:
        pytest.skip("umap not available")

    reducer = umap.UMAP(n_components=3)
    scaler = MinMaxScaler()

    generator = FalseColorGenerator(sample_image_shape, reducer=reducer, scaler=scaler)

    assert generator.reducer is reducer
    assert generator.scaler is scaler
    # When reducer/scaler are provided, they're assumed to be trained
    assert generator.reducer_is_trained
    assert generator.scaler_is_trained


def test_false_color_generator_coordinate_precomputation(sample_image_shape):
    """Test that coordinates are precomputed correctly."""
    generator = FalseColorGenerator(sample_image_shape)

    assert generator.Y is not None
    assert generator.X is not None
    assert generator.Y.shape == (1, 1, 64, 64)
    assert generator.X.shape == (1, 1, 64, 64)
    assert generator.patch_X is not None
    assert generator.patch_Y is not None
    assert generator.mean_patch_X is not None
    assert generator.mean_patch_Y is not None
    assert generator.idx is not None


def test_train_reducer_from_patches(sample_image_shape):
    """Test training reducer from patches."""
    if umap is None:
        pytest.skip("umap not available")

    generator = FalseColorGenerator(sample_image_shape)

    # Create some sample patches
    # Shape should be (N, C, Y, X) where N is number of patches
    patches = torch.randn(10, 1, 32, 32)

    # This should not raise an error
    generator.train_reducer_from_patches(patches)

    assert generator.reducer_is_trained


def test_train_reducer(sample_images, sample_image_shape):
    """Test training reducer from images."""
    if umap is None:
        pytest.skip("umap not available")

    generator = FalseColorGenerator(sample_image_shape)

    # Train with all patches
    generator.train_reducer(sample_images, num_patches=None)

    assert generator.reducer_is_trained
    assert generator.scaler_is_trained


def test_train_reducer_subset_patches(sample_images, sample_image_shape):
    """Test training reducer with subset of patches."""
    if umap is None:
        pytest.skip("umap not available")

    generator = FalseColorGenerator(sample_image_shape)

    # Train with subset of patches
    generator.train_reducer(sample_images, num_patches=5)

    assert generator.reducer_is_trained
    assert generator.scaler_is_trained


def test_call_without_training(sample_image, sample_image_shape):
    """Test __call__ method when reducer/scaler are not trained."""
    if umap is None:
        pytest.skip("umap not available")

    generator = FalseColorGenerator(sample_image_shape)

    # Need to train reducer first - it doesn't auto-train
    # But scaler can auto-train if reducer is trained
    training_images = torch.randn(4, 1, 64, 64)
    generator.train_reducer(training_images, num_patches=10)

    # Now call - scaler should auto-train if needed
    result = generator(sample_image)

    assert result.shape == (64, 64, 3)  # RGB output
    assert result.dtype == np.float64 or result.dtype == np.float32
    assert generator.scaler_is_trained


def test_call_with_trained_reducer_scaler(sample_image, sample_image_shape):
    """Test __call__ method with pre-trained reducer and scaler."""
    if umap is None:
        pytest.skip("umap not available")

    generator = FalseColorGenerator(sample_image_shape)

    # Train first
    training_images = torch.randn(4, 1, 64, 64)
    generator.train_reducer(training_images, num_patches=10)

    # Now call
    result = generator(sample_image)

    assert result.shape == (64, 64, 3)
    # After clipping, all values should be <= 1
    assert np.all(result <= 1.0)


def test_call_clips_values_above_one(sample_image_shape):
    """Test that values above 1 are clipped."""
    if umap is None:
        pytest.skip("umap not available")

    generator = FalseColorGenerator(sample_image_shape)

    # Create a custom scaler that might produce values > 1
    class MockScaler:
        def __init__(self):
            self.is_fitted = False

        def fit_transform(self, X):
            self.is_fitted = True
            # Return values that exceed 1
            return X * 2.0

        def transform(self, X):
            return X * 2.0

    generator.scaler = MockScaler()
    generator.scaler_is_trained = False

    # Train reducer
    training_images = torch.randn(4, 1, 64, 64)
    generator.train_reducer(training_images, num_patches=10)

    # Call with test image
    test_image = torch.randn(1, 1, 64, 64)
    result = generator(test_image)

    # Check that all values are <= 1 (clipped)
    assert np.all(result <= 1.0)
    assert result.shape == (64, 64, 3)


def test_call_asserts_single_image(sample_image_shape):
    """Test that __call__ asserts image has batch size 1."""
    if umap is None:
        pytest.skip("umap not available")

    generator = FalseColorGenerator(sample_image_shape)

    # Train first
    training_images = torch.randn(4, 1, 64, 64)
    generator.train_reducer(training_images, num_patches=10)

    # Try with batch size > 1 - should assert
    multi_image = torch.randn(2, 1, 64, 64)

    with pytest.raises(AssertionError):
        generator(multi_image)


def test_train_reducer_asserts_4d_shape(sample_image_shape):
    """Test that train_reducer asserts 4D input shape."""
    if umap is None:
        pytest.skip("umap not available")

    generator = FalseColorGenerator(sample_image_shape)

    # Try with wrong shape - should assert
    wrong_shape = torch.randn(1, 64, 64)  # 3D instead of 4D

    with pytest.raises(AssertionError):
        generator.train_reducer(wrong_shape)


def test_different_image_shapes():
    """Test FalseColorGenerator with different image shapes."""
    if umap is None:
        pytest.skip("umap not available")

    # Test with different sizes - need enough patches for UMAP
    for H, W in [(128, 128), (64, 128), (64, 64)]:
        image_shape = (1, 1, H, W)
        generator = FalseColorGenerator(image_shape)

        assert generator.image_shape == image_shape
        assert generator.Y.shape == (1, 1, H, W)
        assert generator.X.shape == (1, 1, H, W)

        # Test it works - need enough training data and patches
        test_image = torch.randn(1, 1, H, W)
        training_images = torch.randn(4, 1, H, W)
        # Use None to get all patches, or ensure enough patches
        generator.train_reducer(training_images, num_patches=None)
        result = generator(test_image)

        assert result.shape == (H, W, 3)


def test_different_channel_counts():
    """Test FalseColorGenerator with different channel counts."""
    if umap is None:
        pytest.skip("umap not available")

    # Test with multi-channel input
    image_shape = (1, 3, 64, 64)  # RGB input
    generator = FalseColorGenerator(image_shape)

    test_image = torch.randn(1, 3, 64, 64)
    training_images = torch.randn(2, 3, 64, 64)
    generator.train_reducer(training_images, num_patches=5)
    result = generator(test_image)

    assert result.shape == (64, 64, 3)  # Still RGB output


def test_coordinate_interpolation():
    """Test that coordinate interpolation works correctly."""
    generator = FalseColorGenerator((1, 1, 64, 64))

    # Check that idx is computed and has correct shape
    assert generator.idx is not None
    assert len(generator.idx) == 64 * 64  # One index per pixel

    # Check that all indices are valid (non-negative)
    assert np.all(generator.idx >= 0)


def test_patch_coordinate_mean(sample_image_shape):
    """Test that patch coordinates are averaged correctly."""
    generator = FalseColorGenerator(sample_image_shape)

    # Mean coordinates - after mean operation, shape is (n_patches, 1)
    # since we keep the channel dimension
    n_patches = generator.patch_X.shape[0]
    assert generator.mean_patch_X.shape == (n_patches, 1)
    assert generator.mean_patch_Y.shape == (n_patches, 1)
