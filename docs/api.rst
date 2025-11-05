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

