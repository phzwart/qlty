"""Top-level package for qlty."""

__author__ = """Petrus H. Zwart"""
__email__ = "PHZwart@lbl.gov"
__version__ = "1.1.0.dev0"

# Import cleanup functions
from qlty.cleanup import (
    weed_sparse_classification_training_pairs_2D,
    weed_sparse_classification_training_pairs_3D,
)

# Import patch pair extraction (2D)
from qlty.patch_pairs_2d import extract_overlapping_pixels, extract_patch_pairs

# Import patch pair extraction (3D)
from qlty.patch_pairs_3d import extract_overlapping_pixels_3d, extract_patch_pairs_3d

# Import main classes from all modules
from qlty.qlty2D import NCYXQuilt
from qlty.qlty2DLarge import LargeNCYXQuilt
from qlty.qlty3D import NCZYXQuilt
from qlty.qlty3DLarge import LargeNCZYXQuilt

# Make all classes and functions available at the top level
__all__ = [
    "NCYXQuilt",
    "NCZYXQuilt",
    "LargeNCYXQuilt",
    "LargeNCZYXQuilt",
    "weed_sparse_classification_training_pairs_2D",
    "weed_sparse_classification_training_pairs_3D",
    "extract_patch_pairs",
    "extract_overlapping_pixels",
    "extract_patch_pairs_3d",
    "extract_overlapping_pixels_3d",
]
