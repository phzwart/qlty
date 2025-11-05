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
    patches1, patches2, deltas = extract_patch_pairs(
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

.. autofunction:: qlty.patch_pairs.extract_patch_pairs_3d

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

.. autofunction:: qlty.patch_pairs.extract_overlapping_pixels_3d

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
