Usage
=====

qlty provides tools to unstitch and stitch PyTorch tensors efficiently.

Basic Import
------------

To use qlty in a project, import it::

    import qlty
    from qlty import NCYXQuilt, NCZYXQuilt
    from qlty import LargeNCYXQuilt, LargeNCZYXQuilt

2D In-Memory Processing
------------------------

Basic Example
~~~~~~~~~~~~~

Let's make some mock data and process it::

    import einops
    import torch
    import numpy as np

    # Create sample data
    x = torch.rand((10, 3, 128, 128))  # Input images: (N, C, Y, X)
    y = torch.rand((10, 1, 128, 128))  # Target labels: (N, C, Y, X)

Assume that x and y are data whose relation you are trying to learn using some network, such that after training, you have::

    y_guess = net(x)

with::

    torch.sum(torch.abs(y_guess - y)) < a_small_number

If the data you have is large and doesn't fit onto your GPU card, or if you need to chop things up into smaller bits for boundary detection, qlty can be used. Let's take the above data and chop it into smaller bits::

    quilt = qlty.NCYXQuilt(
        Y=128,
        X=128,
        window=(16, 16),      # Patch size
        step=(4, 4),          # Step size (creates overlap)
        border=(4, 4),        # Border region
        border_weight=0.1     # Weight for border pixels
    )

This object now allows one to cut any input tensor with shape (N, C, Y, X) into smaller, overlapping patches of size (M, C, Y_window, X_window). The moving window, in this case a 16x16 patch, is moved along the input tensor with steps (4, 4). In addition, we define a border region in these patches of 4 pixels wide. Pixels in this area will be assigned weight border_weight (0.1 in this case) when data is stitched back together.

Unstitching Data Pairs
~~~~~~~~~~~~~~~~~~~~~~~

Let's unstitch the (x, y) training data pair::

    x_bits, y_bits = quilt.unstitch_data_pair(x, y)
    print("x shape: ", x.shape)
    print("y shape: ", y.shape)
    print("x_bits shape:", x_bits.shape)
    print("y_bits shape:", y_bits.shape)

Yielding::

    x shape:  torch.Size([10, 3, 128, 128])
    y shape:  torch.Size([10, 1, 128, 128])
    x_bits shape: torch.Size([8410, 3, 16, 16])
    y_bits shape: torch.Size([8410, 16, 16])

Stitching Back Together
~~~~~~~~~~~~~~~~~~~~~~~~

If we now make some mock data that a neural network has returned::

    y_mock = torch.rand((8410, 17, 16, 16))

we can stitch it back together into the right shape, averaging overlapping areas, excluding or downweighting border areas::

    y_stitched, weights = quilt.stitch(y_mock)

which gives::

    print(y_stitched.shape)
    torch.Size([10, 17, 128, 128])

The 'weights' tensor encodes how many contributors there were for each pixel.

Using Numba Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~

The 2D stitch method can use Numba JIT compilation for faster processing::

    result, weights = quilt.stitch(patches, use_numba=True)  # Default
    result, weights = quilt.stitch(patches, use_numba=False)  # Pure PyTorch

3D Volume Processing
--------------------

For 3D volumes, use NCZYXQuilt::

    import torch
    from qlty import NCZYXQuilt

    # Create 3D quilt object
    quilt = NCZYXQuilt(
        Z=64, Y=64, X=64,
        window=(32, 32, 32),   # 3D patch size
        step=(16, 16, 16),     # Step in Z, Y, X
        border=(4, 4, 4),      # Border in each dimension
        border_weight=0.1
    )

    # Process 3D volume
    volume = torch.randn(5, 1, 64, 64, 64)  # (N, C, Z, Y, X)
    patches = quilt.unstitch(volume)

    # Process patches...
    processed = your_model(patches)

    # Stitch back
    reconstructed, weights = quilt.stitch(processed)

Large Dataset Processing (Disk-Cached)
---------------------------------------

For very large datasets that don't fit in memory, use the Large classes::

    import torch
    import tempfile
    import os
    from qlty import LargeNCYXQuilt

    # Create temporary directory for cache
    temp_dir = tempfile.mkdtemp()
    filename = os.path.join(temp_dir, "my_dataset")

    # Create Large quilt object
    quilt = LargeNCYXQuilt(
        filename=filename,
        N=100,              # Number of images
        Y=512, X=512,       # Image dimensions
        window=(128, 128),
        step=(64, 64),
        border=(10, 10),
        border_weight=0.1
    )

    # Load your data
    data = torch.randn(100, 3, 512, 512)

    # Process all chunks
    for i in range(quilt.N_chunks):
        index, patch = quilt.unstitch_next(data)

        # Process patch (e.g., with neural network)
        processed = your_model(patch.unsqueeze(0))

        # Accumulate result
        quilt.stitch(processed, index)

    # Get final result
    mean_result = quilt.return_mean()
    mean_with_std = quilt.return_mean(std=True)

Handling Missing Data
---------------------

When working with sparse or incomplete data, you can filter out patches with no valid data::

    from qlty import NCYXQuilt, weed_sparse_classification_training_pairs_2D

    quilt = NCYXQuilt(
        Y=128, X=128,
        window=(32, 32),
        step=(16, 16),
        border=(5, 5),
        border_weight=0.1
    )

    # Create data with missing labels
    input_data = torch.randn(10, 3, 128, 128)
    labels = torch.ones(10, 128, 128) * (-1)  # Missing label = -1
    labels[:, 20:108, 20:108] = 1.0            # Some valid data

    # Unstitch with missing label handling
    input_patches, label_patches = quilt.unstitch_data_pair(
        input_data, labels, missing_label=-1
    )

    # Filter out patches with no valid data
    border_tensor = quilt.border_tensor()
    valid_input, valid_labels, mask = weed_sparse_classification_training_pairs_2D(
        input_patches, label_patches, missing_label=-1, border_tensor=border_tensor
    )

    print(f"Original patches: {input_patches.shape[0]}")
    print(f"Valid patches: {valid_input.shape[0]}")

Advanced: Working with Border Regions
--------------------------------------

The border tensor indicates which pixels are in the border region::

    border_mask = quilt.border_tensor()
    print(border_mask.shape)  # (window_height, window_width)
    print(border_mask.sum())  # Number of valid (non-border) pixels

Border regions are set to 0.0, valid regions to 1.0. This can be used to mask out border regions during training.

Computing Chunk Information
----------------------------

To know how many patches will be created::

    nY, nX = quilt.get_times()
    print(f"Patches in Y direction: {nY}")
    print(f"Patches in X direction: {nX}")
    print(f"Total patches per image: {nY * nX}")

For a tensor with N images, the total number of patches will be N * nY * nX.

Best Practices
--------------

1. **Overlap Strategy**:
   - Use step size = window/2 for 50% overlap (common choice)
   - More overlap = smoother results but more computation
   - Less overlap = faster but may have artifacts

2. **Border Size**:
   - Typically 10-20% of window size
   - Larger for networks sensitive to edge effects
   - Smaller for networks with good edge handling

3. **Border Weight**:
   - 0.1 is a good default
   - 0.0 completely excludes borders
   - 1.0 gives equal weight (not recommended)

4. **Memory Management**:
   - Use in-memory classes (NCYXQuilt, NCZYXQuilt) if data fits in RAM
   - Use Large classes for datasets > several GB
   - Large classes use Zarr for efficient disk caching

5. **Softmax Warning**:
   - Apply softmax AFTER stitching, not before
   - Averaging softmaxed tensors ≠ softmax of averaged tensors
   - Process logits, then apply softmax to final result

Common Patterns
---------------

Training Loop Pattern
~~~~~~~~~~~~~~~~~~~~~

::

    quilt = NCYXQuilt(Y=256, X=256, window=(64, 64), step=(32, 32), border=(8, 8))

    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # Unstitch
            img_patches, lbl_patches = quilt.unstitch_data_pair(images, labels)

            # Train
            for img, lbl in zip(img_patches, lbl_patches):
                output = model(img.unsqueeze(0))
                loss = criterion(output, lbl.unsqueeze(0))
                # ...

Inference Pattern
~~~~~~~~~~~~~~~~~

::

    quilt = NCYXQuilt(Y=512, X=512, window=(128, 128), step=(64, 64), border=(10, 10))

    # Unstitch
    patches = quilt.unstitch(test_image)

    # Process
    with torch.no_grad():
        outputs = model(patches)

    # Stitch
    result, weights = quilt.stitch(outputs)

Large Dataset Pattern
~~~~~~~~~~~~~~~~~~~~~

::

    quilt = LargeNCYXQuilt(filename, N=1000, Y=1024, X=1024,
                          window=(256, 256), step=(128, 128), border=(20, 20))

    # Process in chunks
    for i in range(quilt.N_chunks):
        idx, patch = quilt.unstitch_next(data)
        processed = model(patch.unsqueeze(0))
        quilt.stitch(processed, idx)

    # Get results
    mean = quilt.return_mean()
    mean, std = quilt.return_mean(std=True)

Patch Pair Extraction
---------------------

The patch pair extraction feature allows you to extract pairs of patches from images with controlled displacement between them. This is useful for training models that learn relationships between nearby image regions, such as self-supervised learning, contrastive learning, or learning spatial correspondences.

Overview
~~~~~~~~

The patch pair extraction works by:

1. **Extracting patch pairs**: For each image, randomly samples pairs of patches where the displacement between patch centers follows a specified Euclidean distance constraint.

2. **Finding overlapping regions**: Given the displacement vectors, identifies which pixels in the two patches correspond to the same spatial location in the original image.

3. **Gradient-friendly**: All operations preserve gradients, making it suitable for end-to-end training.

Basic Usage
~~~~~~~~~~~

Extract patch pairs from a tensor::

    import torch
    from qlty import extract_patch_pairs, extract_overlapping_pixels

    # Create input tensor: (N, C, Y, X)
    tensor = torch.randn(10, 3, 128, 128)

    # Extract patch pairs
    window = (32, 32)  # 32x32 patches
    num_patches = 5    # 5 patch pairs per image
    delta_range = (8.0, 16.0)  # Euclidean distance between 8 and 16 pixels

    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # patches1: (50, 3, 32, 32) - first patches
    # patches2: (50, 3, 32, 32) - second patches (displaced)
    # deltas: (50, 2) - displacement vectors (dx, dy)

Extract overlapping pixels::

    # Get overlapping pixels from patch pairs
    overlapping1, overlapping2 = extract_overlapping_pixels(
        patches1, patches2, deltas
    )

    # overlapping1: (K, 3) - overlapping pixels from patches1
    # overlapping2: (K, 3) - overlapping pixels from patches2
    # K is the total number of overlapping pixels across all pairs
    # Corresponding pixels are at the same index in both tensors

Delta Range Constraints
~~~~~~~~~~~~~~~~~~~~~~~

The ``delta_range`` parameter specifies the Euclidean distance constraint for displacement vectors:

- **Constraint**: ``low <= sqrt(dx² + dy²) <= high``
- **Range requirement**: ``window//4 <= low <= high <= 3*window//4``
  where ``window`` is the maximum of patch height and width

This ensures that:
- Displacements are not too small (patches would be nearly identical)
- Displacements are not too large (patches would have no overlap)
- There's meaningful overlap for learning correspondences

Example: For a 32x32 window, valid delta_range is approximately (8, 24).

Use Case: Kernel Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common use case is optimizing neural network kernels using L1 loss on overlapping pixels::

    import torch
    import torch.nn as nn
    from qlty import extract_patch_pairs, extract_overlapping_pixels

    # Create input tensor
    tensor = torch.randn(10, 1, 32, 32)

    # Extract patch pairs
    patches1, patches2, deltas = extract_patch_pairs(
        tensor, window=(9, 9), num_patches=5, delta_range=(3.0, 6.0)
    )
    patches1 = patches1.detach()
    patches2 = patches2.detach()

    # Create two Conv2D layers
    conv1 = nn.Conv2d(1, 1, 3, padding=1, padding_mode='reflect', bias=False)
    conv2 = nn.Conv2d(1, 1, 3, padding=1, padding_mode='reflect', bias=False)

    # Optimize conv2 to match conv1 using L1 loss on overlapping pixels
    optimizer = torch.optim.Adam(conv2.parameters(), lr=0.05)

    for iteration in range(100):
        optimizer.zero_grad()

        output1 = conv1(patches1)
        output2 = conv2(patches2)

        # Extract overlapping pixels
        overlapping1, overlapping2 = extract_overlapping_pixels(
            output1, output2, deltas
        )

        # L1 loss on corresponding pixels
        loss = torch.nn.functional.l1_loss(overlapping1, overlapping2)
        loss.backward()
        optimizer.step()

How It Works
~~~~~~~~~~~

**Patch Pair Extraction:**

1. For each image in the input tensor, randomly samples ``num_patches`` locations.
2. For each location ``(x_i, y_i)``, samples a displacement vector ``(dx_i, dy_i)`` such that the Euclidean distance ``sqrt(dx_i² + dy_i²)`` is within ``delta_range``.
3. Extracts two patches:
   - Patch 1 at ``(x_i, y_i)``
   - Patch 2 at ``(x_i + dx_i, y_i + dy_i)``
4. Ensures both patches fit within image boundaries.

**Overlapping Pixel Extraction:**

1. For each patch pair, computes which pixels have valid correspondences:
   - A pixel at ``(u1, v1)`` in patch1 corresponds to ``(u1 - dy, v1 - dx)`` in patch2
   - Only pixels where both coordinates are valid (within patch bounds) are included
2. Extracts the overlapping regions from both patches.
3. Flattens and concatenates all overlapping pixels into ``(K, C)`` tensors.

**Key Properties:**

- **Partial overlap**: Typically 30-70% of pixels overlap, depending on displacement
- **Gradient preservation**: All operations maintain the computation graph
- **GPU-friendly**: Optimized for GPU execution with minimal CPU-GPU transfers
- **Reproducible**: Optional random seed for consistent results

Mathematical Details
~~~~~~~~~~~~~~~~~~~~

Given:
- Patch 1 extracted at ``(x, y)`` with size ``(U, V)``
- Patch 2 extracted at ``(x + dx, y + dy)`` with size ``(U, V)``
- Displacement vector ``(dx, dy)``

A pixel at ``(u1, v1)`` in patch1 corresponds to the same spatial location as pixel ``(u2, v2)`` in patch2 when:

- ``u2 = u1 - dy``
- ``v2 = v1 - dx``

The overlap region in patch1 coordinates is:
- ``u1 in [max(0, dy), min(U, U + dy))``
- ``v1 in [max(0, dx), min(V, V + dx))``

This ensures both corresponding pixels are within their respective patch bounds.

3D Patch Pair Extraction
-------------------------

The same functionality is available for 3D volumes (volumetric data).

Basic Usage (3D)
~~~~~~~~~~~~~~~~

Extract patch pairs from a 3D tensor::

    import torch
    from qlty import extract_patch_pairs_3d, extract_overlapping_pixels_3d

    # Create input tensor: (N, C, Z, Y, X)
    tensor = torch.randn(5, 1, 64, 64, 64)  # 5 volumes, 1 channel, 64x64x64

    # Extract patch pairs
    window = (16, 16, 16)  # 16x16x16 patches
    num_patches = 10  # 10 patch pairs per volume
    delta_range = (8.0, 12.0)  # Euclidean distance between 8 and 12 voxels

    patches1, patches2, deltas = extract_patch_pairs_3d(
        tensor, window, num_patches, delta_range, random_seed=42
    )

    # patches1: (50, 1, 16, 16, 16) - first patches
    # patches2: (50, 1, 16, 16, 16) - second patches (displaced)
    # deltas: (50, 3) - displacement vectors (dx, dy, dz)

Extract overlapping pixels from 3D patches::

    # Get overlapping pixels from patch pairs
    overlapping1, overlapping2 = extract_overlapping_pixels_3d(
        patches1, patches2, deltas
    )

    # overlapping1: (K, 1) - overlapping pixels from patches1
    # overlapping2: (K, 1) - overlapping pixels from patches2
    # K is the total number of overlapping pixels across all pairs

3D Mathematical Details
~~~~~~~~~~~~~~~~~~~~~~~

Given:
- Patch 1 extracted at ``(x, y, z)`` with size ``(U, V, W)``
- Patch 2 extracted at ``(x + dx, y + dy, z + dz)`` with size ``(U, V, W)``
- Displacement vector ``(dx, dy, dz)``

A pixel at ``(u1, v1, w1)`` in patch1 corresponds to the same spatial location as pixel ``(u2, v2, w2)`` in patch2 when:

- ``u2 = u1 - dz``
- ``v2 = v1 - dy``
- ``w2 = w1 - dx``

The overlap region in patch1 coordinates is:
- ``u1 in [max(0, dz), min(U, U + dz))``
- ``v1 in [max(0, dy), min(V, V + dy))``
- ``w1 in [max(0, dx), min(W, W + dx))``

The Euclidean distance constraint is: ``low <= sqrt(dx² + dy² + dz²) <= high``
