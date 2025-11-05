"""Top-level package for qlty."""

__author__ = """Petrus H. Zwart"""
__email__ = "PHZwart@lbl.gov"
__version__ = "0.3.0"

# Import cleanup functions
from .cleanup import (weed_sparse_classification_training_pairs_2D,
                      weed_sparse_classification_training_pairs_3D)
# Import main classes from all modules
from .qlty2D import NCYXQuilt
from .qlty2DLarge import LargeNCYXQuilt
from .qlty3D import NCZYXQuilt
from .qlty3DLarge import LargeNCZYXQuilt

# Make all classes and functions available at the top level
__all__ = [
    "NCYXQuilt",
    "NCZYXQuilt",
    "LargeNCYXQuilt",
    "LargeNCZYXQuilt",
    "weed_sparse_classification_training_pairs_2D",
    "weed_sparse_classification_training_pairs_3D",
]
