API Reference
==============

This page provides detailed API documentation for all public classes and functions in qlty.

In-Memory Classes
------------------

NCYXQuilt
~~~~~~~~~~

.. autoclass:: qlty.qlty2D.NCYXQuilt
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: python

    from qlty import NCYXQuilt

    quilt = NCYXQuilt(
        Y=128, X=128,
        window=(32, 32),
        step=(16, 16),
        border=(5, 5),
        border_weight=0.1
    )

    data = torch.randn(10, 3, 128, 128)
    patches = quilt.unstitch(data)
    reconstructed, weights = quilt.stitch(patches)

NCZYXQuilt
~~~~~~~~~~

.. autoclass:: qlty.qlty3D.NCZYXQuilt
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: python

    from qlty import NCZYXQuilt

    quilt = NCZYXQuilt(
        Z=64, Y=64, X=64,
        window=(32, 32, 32),
        step=(16, 16, 16),
        border=(4, 4, 4),
        border_weight=0.1
    )

    volume = torch.randn(5, 1, 64, 64, 64)
    patches = quilt.unstitch(volume)
    reconstructed, weights = quilt.stitch(patches)

Disk-Cached Classes
--------------------

LargeNCYXQuilt
~~~~~~~~~~~~~~

.. autoclass:: qlty.qlty2DLarge.LargeNCYXQuilt
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: python

    from qlty import LargeNCYXQuilt
    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()
    filename = os.path.join(temp_dir, "dataset")

    quilt = LargeNCYXQuilt(
        filename=filename,
        N=100,
        Y=512, X=512,
        window=(128, 128),
        step=(64, 64),
        border=(10, 10),
        border_weight=0.1
    )

    data = torch.randn(100, 3, 512, 512)
    for i in range(quilt.N_chunks):
        idx, patch = quilt.unstitch_next(data)
        processed = model(patch.unsqueeze(0))
        quilt.stitch(processed, idx)

    result = quilt.return_mean()

LargeNCZYXQuilt
~~~~~~~~~~~~~~~

.. autoclass:: qlty.qlty3DLarge.LargeNCZYXQuilt
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
------------------

weed_sparse_classification_training_pairs_2D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.cleanup.weed_sparse_classification_training_pairs_2D

**Example:**

.. code-block:: python

    from qlty import NCYXQuilt, weed_sparse_classification_training_pairs_2D

    quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5))

    input_patches = torch.randn(100, 3, 32, 32)
    label_patches = torch.ones(100, 32, 32) * (-1)  # Missing labels
    label_patches[0:50] = 1.0  # Some valid

    border_tensor = quilt.border_tensor()
    valid_in, valid_out, mask = weed_sparse_classification_training_pairs_2D(
        input_patches, label_patches, missing_label=-1, border_tensor=border_tensor
    )

weed_sparse_classification_training_pairs_3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.cleanup.weed_sparse_classification_training_pairs_3D

Patch Pair Extraction Functions
---------------------------------

extract_patch_pairs
~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.patch_pairs_2d.extract_patch_pairs

**Example:**

.. code-block:: python

    from qlty import extract_patch_pairs
    import torch

    tensor = torch.randn(5, 3, 128, 128)  # 5 images, 3 channels, 128x128
    window = (32, 32)  # 32x32 patches
    num_patches = 10  # 10 patch pairs per image
    delta_range = (8.0, 16.0)  # Euclidean distance between 8 and 16 pixels

    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # patches1: (50, 3, 32, 32) - patches at original locations
    # patches2: (50, 3, 32, 32) - patches at displaced locations
    # deltas: (50, 2) - displacement vectors (dx, dy)

extract_overlapping_pixels
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.patch_pairs_2d.extract_overlapping_pixels

**Example:**

.. code-block:: python

    from qlty import extract_patch_pairs, extract_overlapping_pixels
    import torch

    # Extract patch pairs
    patches1, patches2, deltas, rotations = extract_patch_pairs(
    patches1, patches2, deltas, rotations = extract_patch_pairs(
        tensor, window=(32, 32), num_patches=10, delta_range=(8.0, 16.0)
    )

    # Extract overlapping pixels
    overlapping1, overlapping2 = extract_overlapping_pixels(
        patches1, patches2, deltas
    )

    # overlapping1: (K, 3) - overlapping pixels from patches1
    # overlapping2: (K, 3) - overlapping pixels from patches2
    # K is the total number of overlapping pixels
    # Corresponding pixels are at the same index in both tensors

extract_patch_pairs_3d
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.patch_pairs_3d.extract_patch_pairs_3d

**Example:**

.. code-block:: python

    from qlty import extract_patch_pairs_3d
    import torch

    tensor = torch.randn(5, 1, 64, 64, 64)  # 5 volumes, 1 channel, 64x64x64
    window = (16, 16, 16)  # 16x16x16 patches
    num_patches = 10  # 10 patch pairs per volume
    delta_range = (8.0, 12.0)  # Euclidean distance between 8 and 12 voxels

    patches1, patches2, deltas = extract_patch_pairs_3d(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # patches1: (50, 1, 16, 16, 16) - patches at original locations
    # patches2: (50, 1, 16, 16, 16) - patches at displaced locations
    # deltas: (50, 3) - displacement vectors (dx, dy, dz)

extract_overlapping_pixels_3d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.patch_pairs_3d.extract_overlapping_pixels_3d

**Example:**

.. code-block:: python

    from qlty import extract_patch_pairs_3d, extract_overlapping_pixels_3d
    import torch

    # Extract patch pairs
    patches1, patches2, deltas = extract_patch_pairs_3d(
        tensor, window=(16, 16, 16), num_patches=10, delta_range=(8.0, 12.0)
    )

    # Extract overlapping pixels
    overlapping1, overlapping2 = extract_overlapping_pixels_3d(
        patches1, patches2, deltas
    )

    # overlapping1: (K, 1) - overlapping pixels from patches1
    # overlapping2: (K, 1) - overlapping pixels from patches2
    # K is the total number of overlapping pixels
    # Corresponding pixels are at the same index in both tensors

Advanced Patch Pair Functions
-------------------------------

extract_patch_pairs_metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.patch_pairs_2d.extract_patch_pairs_metadata

**Example:**

.. code-block:: python

    from qlty.patch_pairs_2d import extract_patch_pairs_metadata
    import torch

    tensor = torch.randn(10, 3, 128, 128)  # 10 images, 3 channels, 128x128
    window = (32, 32)
    num_patches = 20
    delta_range = (8.0, 16.0)

    # Extract metadata without loading patches into memory
    metadata = extract_patch_pairs_metadata(
        tensor, window, num_patches, delta_range,
        random_seed=42, num_workers=4
    )

    # metadata contains:
    # - patch1_x, patch1_y: Coordinates of first patches
    # - patch2_x, patch2_y: Coordinates of second patches
    # - dx, dy: Displacement vectors
    # - rotation: Rotation applied to second patch
    # - image_idx: Which image each patch pair came from
    # - mean1, mean2: Mean values of patches
    # - sigma1, sigma2: Standard deviations of patches
    # - window: Window size used

extract_patches_from_metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.patch_pairs_2d.extract_patches_from_metadata

**Example:**

.. code-block:: python

    from qlty.patch_pairs_2d import (
        extract_patch_pairs_metadata,
        extract_patches_from_metadata
    )
    import torch

    tensor = torch.randn(10, 3, 128, 128)
    window = (32, 32)
    num_patches = 20
    delta_range = (8.0, 16.0)

    # First, extract metadata
    metadata = extract_patch_pairs_metadata(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # Later, extract patches for specific indices
    selected_indices = [0, 5, 10, 15]  # Extract only these patch pairs
    patches1, patches2, deltas, rotations = extract_patches_from_metadata(
        tensor, metadata, selected_indices
    )

    # patches1: (4, 3, 32, 32) - only selected patches
    # patches2: (4, 3, 32, 32)
    # deltas: (4, 2)
    # rotations: (4,)

extract_patches_to_zarr
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.patch_pairs_2d.extract_patches_to_zarr

**Example:**

.. code-block:: python

    from qlty.patch_pairs_2d import (
        extract_patch_pairs_metadata,
        extract_patches_to_zarr
    )
    import torch
    import zarr

    tensor = torch.randn(10, 3, 128, 128)
    window = (32, 32)
    num_patches = 20
    delta_range = (8.0, 16.0)

    # Extract metadata
    metadata = extract_patch_pairs_metadata(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # Save patches to Zarr format
    zarr_path = "patches.zarr"
    group = zarr.open_group(zarr_path, mode="w")

    extract_patches_to_zarr(
        tensor, metadata, group,
        chunk_size=(100, 3, 32, 32)  # Custom chunk size
    )

    # Patches are now stored in Zarr format:
    # - group["patches1"]: (N*num_patches, C, U, V)
    # - group["patches2"]: (N*num_patches, C, U, V)
    # - group["deltas"]: (N*num_patches, 2)
    # - group["rotations"]: (N*num_patches,)
    # - group.attrs["metadata"]: Original metadata dict

ZarrPatchPairDataset
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qlty.patch_pairs_2d.ZarrPatchPairDataset
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: python

    from qlty.patch_pairs_2d import ZarrPatchPairDataset
    from torch.utils.data import DataLoader
    import zarr

    # Open existing Zarr group with patches
    zarr_path = "patches.zarr"
    group = zarr.open_group(zarr_path, mode="r")

    # Create PyTorch Dataset
    dataset = ZarrPatchPairDataset(group)

    # Use with DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for patches1, patches2, deltas, rotations in dataloader:
        # patches1: (batch_size, C, U, V)
        # patches2: (batch_size, C, U, V)
        # deltas: (batch_size, 2)
        # rotations: (batch_size,)
        # Train your model...
        pass

    # With custom transform
    def normalize_patches(p1, p2, d, r):
        p1 = (p1 - p1.mean()) / p1.std()
        p2 = (p2 - p2.mean()) / p2.std()
        return p1, p2, d, r

    dataset = ZarrPatchPairDataset(group, transform=normalize_patches)

Image Stack Utilities
----------------------

stack_files_to_zarr
~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.utils.stack_to_zarr.stack_files_to_zarr

**Example:**

.. code-block:: python

    from qlty.utils.stack_to_zarr import stack_files_to_zarr
    from pathlib import Path

    # Convert image stack to Zarr format
    result = stack_files_to_zarr(
        directory="/path/to/images",
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",  # Matches: stack_001.tif, stack_002.tif
        axis_order="CZYX"  # Channel, Z, Y, X
    )

    # result is a dict with metadata:
    # {
    #     "stack": {
    #         "zarr_path": "/path/to/images/stack.zarr",
    #         "shape": (100, 3, 512, 512),  # (Z, C, Y, X)
    #         "file_count": 100,
    #         "axis_order": "CZYX"
    #     }
    # }

stack_files_to_ome_zarr
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.utils.stack_to_zarr.stack_files_to_ome_zarr

**Example:**

.. code-block:: python

    from qlty.utils.stack_to_zarr import stack_files_to_ome_zarr
    from pathlib import Path

    # Convert image stack to OME-Zarr format with multiscale pyramids
    result = stack_files_to_ome_zarr(
        directory="/path/to/images",
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",
        axis_order="ZCYX",  # Z, Channel, Y, X
        pyramid_levels=4,   # Create 4 resolution levels
        downsample_mode="2d",  # Downsample in 2D (per slice)
        downsample_method="dask"  # Use dask for downsampling
    )

    # result contains metadata and zarr path
    # The OME-Zarr file can be opened with:
    # import zarr
    # group = zarr.open_group(result["stack"]["zarr_path"], mode="r")
    # level_0 = group["0"]  # Full resolution
    # level_1 = group["1"]  # 2x downsampled
    # level_2 = group["2"]  # 4x downsampled
    # etc.

Pre-Tokenization for Patch Processing (2D)
--------------------------------------------

tokenize_patch
~~~~~~~~~~~~~~

.. autofunction:: qlty.pretokenizer_2d.sequences.tokenize_patch

**Example:**

.. code-block:: python

    from qlty import tokenize_patch
    import torch

    # Tokenize a single patch into overlapping subpatches
    patch = torch.randn(3, 64, 64)  # 3 channels, 64x64 patch
    tokens, coords = tokenize_patch(patch, patch_size=16, stride=8)

    # tokens: (T, 768) - flattened token vectors (T tokens, each 3*16*16=768 dims)
    # coords: (T, 2) - absolute (y, x) coordinates of each token
    print(f"Created {tokens.shape[0]} tokens from patch")

build_sequence_pair
~~~~~~~~~~~~~~~~~~~

.. autofunction:: qlty.pretokenizer_2d.sequences.build_sequence_pair

**Example:**

.. code-block:: python

    from qlty import build_sequence_pair, extract_patch_pairs
    import torch

    # Extract patch pairs using qlty's extract_patch_pairs
    images = torch.randn(10, 3, 128, 128)
    patches1, patches2, deltas, rotations = extract_patch_pairs(
        images, window=(64, 64), num_patches=5, delta_range=(10.0, 20.0)
    )

    # Build sequence pairs with overlap information
    # Process a single pair
    result = build_sequence_pair(
        patches1[0],      # (3, 64, 64)
        patches2[0],      # (3, 64, 64)
        dx=deltas[0, 0].item(),
        dy=deltas[0, 1].item(),
        rot_k90=rotations[0].item(),
        patch_size=16,
        stride=8
    )

    # result contains:
    # - tokens1, tokens2: Token sequences from each patch
    # - coords1, coords2: Absolute coordinates for each token
    # - overlap_mask1, overlap_mask2: Which tokens have overlaps
    # - overlap_indices1_to_2, overlap_indices2_to_1: Token mappings
    # - overlap_fractions: Fraction of overlap for each token
    # - overlap_pairs: List of (i, j) pairs of overlapping tokens

    # Process a batch efficiently
    batch_result = build_sequence_pair(
        patches1,         # (50, 3, 64, 64)
        patches2,         # (50, 3, 64, 64)
        dx=deltas[:, 0],  # (50,)
        dy=deltas[:, 1],  # (50,)
        rot_k90=rotations, # (50,)
        patch_size=16,
        stride=8
    )

    # Batch result has same keys but tensors are padded to max length
    # - tokens1, tokens2: (50, T_max, D)
    # - sequence_lengths: (50,) - actual lengths
    # - overlap_pair_counts: (50,) - number of overlaps per pair

Parameter Details
-----------------

Window and Step Sizes
~~~~~~~~~~~~~~~~~~~~~~

- **window**: Size of each patch in pixels
  - 2D: `(Y_size, X_size)`
  - 3D: `(Z_size, Y_size, X_size)`

- **step**: Distance the window moves between patches
  - 2D: `(Y_step, X_step)`
  - 3D: `(Z_step, Y_step, X_step)`
  - Common: step = window/2 for 50% overlap

Border Parameters
~~~~~~~~~~~~~~~~~

- **border**: Size of border region to downweight
  - Can be `int` (same for all dimensions) or `tuple` (per dimension)
  - `None` or `0` means no border
  - Typically 10-20% of window size

- **border_weight**: Weight for border pixels (0.0 to 1.0)
  - 0.0: Completely exclude borders
  - 0.1: Recommended default
  - 1.0: Full weight (not recommended)

Return Types
------------

All methods return PyTorch tensors (in-memory classes) or NumPy arrays (Large classes):

- **unstitch()**: Returns `torch.Tensor` of shape `(M, C, ...)`
- **stitch()**: Returns `Tuple[torch.Tensor, torch.Tensor]` (result, weights)
- **border_tensor()**: Returns `torch.Tensor` (in-memory) or `np.ndarray` (Large)
- **get_times()**: Returns `Tuple[int, ...]` with number of patches per dimension
