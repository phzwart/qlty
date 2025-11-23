"""Top-level package for qlty."""

__author__ = """Petrus H. Zwart"""
__email__ = "PHZwart@lbl.gov"
__version__ = "1.3.0"

# Import cleanup functions
from qlty.cleanup import (
    weed_sparse_classification_training_pairs_2D,
    weed_sparse_classification_training_pairs_3D,
)

# Import patch pair extraction (2D)
from qlty.patch_pairs_2d import extract_overlapping_pixels, extract_patch_pairs

# Import patch pair extraction (3D)
from qlty.patch_pairs_3d import extract_overlapping_pixels_3d, extract_patch_pairs_3d

# Import pre-tokenization utilities (2D)
from qlty.pretokenizer_2d import build_sequence_pair, tokenize_patch

# Import main classes from all modules
from qlty.qlty2D import NCYXQuilt
from qlty.qlty2DLarge import LargeNCYXQuilt
from qlty.qlty3D import NCZYXQuilt
from qlty.qlty3DLarge import LargeNCZYXQuilt

# Import 2.5D quilt
try:
    from qlty.qlty2_5D import NCZYX25DQuilt, ZOperation

    _HAS_2_5D = True
except ImportError:
    _HAS_2_5D = False

# Import backends (optional)
try:
    from qlty.backends_2_5D import (
        DataSource3DBackend,
        HDF5Backend,
        InMemoryBackend,
        MemoryMappedBackend,
        TensorLike3D,
        ZarrBackend,
        from_hdf5,
        from_memmap,
        from_zarr,
    )

    _HAS_BACKENDS = True
except ImportError:
    _HAS_BACKENDS = False

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
    "tokenize_patch",
    "build_sequence_pair",
]

# Add 2.5D exports if available
if _HAS_2_5D:
    __all__.extend(["NCZYX25DQuilt", "ZOperation"])

if _HAS_BACKENDS:
    __all__.extend(
        [
            "DataSource3DBackend",
            "InMemoryBackend",
            "MemoryMappedBackend",
            "ZarrBackend",
            "HDF5Backend",
            "TensorLike3D",
            "from_zarr",
            "from_hdf5",
            "from_memmap",
        ]
    )
