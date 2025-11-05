Examples
========

This page contains comprehensive examples for common use cases.

Example 1: Basic 2D Image Processing
--------------------------------------

Complete workflow for processing 2D images::

    import torch
    from qlty import NCYXQuilt

    # Setup
    quilt = NCYXQuilt(
        Y=256, X=256,
        window=(64, 64),
        step=(32, 32),      # 50% overlap
        border=(8, 8),
        border_weight=0.1
    )

    # Load data
    images = torch.randn(20, 3, 256, 256)

    # Split into patches
    patches = quilt.unstitch(images)
    print(f"Created {patches.shape[0]} patches from {images.shape[0]} images")

    # Process patches
    processed_patches = your_model(patches)

    # Stitch back together
    reconstructed, weights = quilt.stitch(processed_patches)
    assert reconstructed.shape[0] == images.shape[0]

Example 2: Training with Input-Output Pairs
--------------------------------------------

Training a model on unstitched patches::

    from qlty import NCYXQuilt
    import torch

    quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5))

    # Training data
    input_images = torch.randn(100, 3, 128, 128)
    target_labels = torch.randn(100, 128, 128)

    # Unstitch pairs
    input_patches, target_patches = quilt.unstitch_data_pair(input_images, target_labels)

    # Training loop
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    for inp, tgt in zip(input_patches, target_patches):
        optimizer.zero_grad()
        output = model(inp.unsqueeze(0))
        loss = criterion(output, tgt.unsqueeze(0))
        loss.backward()
        optimizer.step()

Example 3: Large Dataset with Disk Caching
--------------------------------------------

Processing datasets too large for memory::

    from qlty import LargeNCYXQuilt
    import torch
    import tempfile
    import os

    # Setup
    temp_dir = tempfile.mkdtemp()
    filename = os.path.join(temp_dir, "large_dataset")

    quilt = LargeNCYXQuilt(
        filename=filename,
        N=1000,            # 1000 images
        Y=1024, X=1024,   # Large images
        window=(256, 256),
        step=(128, 128),
        border=(20, 20),
        border_weight=0.1
    )

    # Load data (or iterate through dataset)
    data = torch.randn(1000, 3, 1024, 1024)

    # Process all chunks
    print(f"Processing {quilt.N_chunks} chunks...")
    for i in range(quilt.N_chunks):
        if i % 100 == 0:
            print(f"Progress: {i}/{quilt.N_chunks}")

        index, patch = quilt.unstitch_next(data)

        # Process patch
        with torch.no_grad():
            processed = model(patch.unsqueeze(0))

        # Accumulate
        quilt.stitch(processed, index)

    # Get final results
    mean_result = quilt.return_mean()
    mean_result, std_result = quilt.return_mean(std=True)

    print(f"Final shape: {mean_result.shape}")

    # Cleanup
    for suffix in ["_mean_cache.zarr", "_std_cache.zarr", "_norma_cache.zarr",
                   "_mean.zarr", "_std.zarr"]:
        path = filename + suffix
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)

Example 4: Handling Sparse/Missing Data
----------------------------------------

Filtering out patches with no valid data::

    from qlty import NCYXQuilt, weed_sparse_classification_training_pairs_2D

    quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5))

    # Data with missing labels
    input_data = torch.randn(50, 3, 128, 128)
    labels = torch.ones(50, 128, 128) * (-1)  # All missing initially

    # Add some valid data
    labels[:, 30:98, 30:98] = torch.randint(0, 10, (50, 68, 68)).float()

    # Unstitch
    input_patches, label_patches = quilt.unstitch_data_pair(
        input_data, labels, missing_label=-1
    )

    print(f"Total patches: {input_patches.shape[0]}")

    # Filter valid patches
    border_tensor = quilt.border_tensor()
    valid_input, valid_labels, removed_mask = weed_sparse_classification_training_pairs_2D(
        input_patches, label_patches, missing_label=-1, border_tensor=border_tensor
    )

    print(f"Valid patches: {valid_input.shape[0]}")
    print(f"Removed patches: {removed_mask.sum().item()}")

Example 5: 3D Volume Processing
--------------------------------

Processing 3D medical imaging or microscopy data::

    from qlty import NCZYXQuilt
    import torch

    quilt = NCZYXQuilt(
        Z=128, Y=128, X=128,
        window=(64, 64, 64),
        step=(32, 32, 32),   # 50% overlap in each dimension
        border=(8, 8, 8),
        border_weight=0.1
    )

    # 3D volume data
    volumes = torch.randn(10, 1, 128, 128, 128)  # (N, C, Z, Y, X)

    # Process
    patches = quilt.unstitch(volumes)
    print(f"Created {patches.shape[0]} patches from {volumes.shape[0]} volumes")

    # Process with 3D model
    processed = your_3d_model(patches)

    # Stitch back
    reconstructed, weights = quilt.stitch(processed)
    assert reconstructed.shape == volumes.shape

Example 6: Inference with Softmax Handling
-------------------------------------------

Correct way to handle softmax when stitching::

    from qlty import NCYXQuilt
    import torch.nn.functional as F

    quilt = NCYXQuilt(Y=256, X=256, window=(64, 64), step=(32, 32), border=(8, 8))

    image = torch.randn(1, 3, 256, 256)
    patches = quilt.unstitch(image)

    # Process patches (get logits, NOT softmax)
    with torch.no_grad():
        logits = model(patches)  # Shape: (M, num_classes, 64, 64)

    # Stitch logits first
    stitched_logits, weights = quilt.stitch(logits)

    # THEN apply softmax
    probabilities = F.softmax(stitched_logits, dim=1)

    # This is correct! Averaging logits then softmaxing = softmax of averaged logits

Example 7: Custom Border Weighting
-----------------------------------

Experimenting with different border weights::

    from qlty import NCYXQuilt

    # Test different border weights
    for border_weight in [0.0, 0.1, 0.5, 1.0]:
        quilt = NCYXQuilt(
            Y=128, X=128,
            window=(32, 32),
            step=(16, 16),
            border=(5, 5),
            border_weight=border_weight
        )

        data = torch.randn(5, 3, 128, 128)
        patches = quilt.unstitch(data)
        reconstructed, weights = quilt.stitch(patches)

        # Evaluate reconstruction quality
        error = torch.mean(torch.abs(reconstructed - data))
        print(f"Border weight {border_weight}: Error = {error:.6f}")

Example 8: Batch Processing for Efficiency
-------------------------------------------

Processing patches in batches for better GPU utilization::

    from qlty import NCYXQuilt
    import torch

    quilt = NCYXQuilt(Y=512, X=512, window=(128, 128), step=(64, 64), border=(10, 10))

    image = torch.randn(1, 3, 512, 512)
    patches = quilt.unstitch(image)

    # Process in batches
    batch_size = 32
    processed_patches = []

    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        with torch.no_grad():
            output = model(batch)
        processed_patches.append(output)

    processed_patches = torch.cat(processed_patches, dim=0)
    result, weights = quilt.stitch(processed_patches)

Example 9: Combining with DataLoaders
--------------------------------------

Integrating with PyTorch DataLoaders::

    from torch.utils.data import Dataset, DataLoader
    from qlty import NCYXQuilt

    class PatchedDataset(Dataset):
        def __init__(self, images, labels, quilt):
            self.quilt = quilt
            self.input_patches, self.label_patches = quilt.unstitch_data_pair(
                images, labels
            )

        def __len__(self):
            return len(self.input_patches)

        def __getitem__(self, idx):
            return self.input_patches[idx], self.label_patches[idx]

    # Create dataset
    images = torch.randn(100, 3, 128, 128)
    labels = torch.randn(100, 128, 128)
    quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16), border=(5, 5))

    dataset = PatchedDataset(images, labels, quilt)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Train
    for batch_input, batch_labels in dataloader:
        # Training code...
        pass

Example 10: Error Handling and Validation
------------------------------------------

Proper error handling::

    from qlty import NCYXQuilt
    import torch

    # Valid usage
    try:
        quilt = NCYXQuilt(
            Y=128, X=128,
            window=(32, 32),
            step=(16, 16),
            border=(5, 5),
            border_weight=0.1
        )
        print("✓ Quilt created successfully")
    except ValueError as e:
        print(f"✗ Error: {e}")

    # Invalid border_weight
    try:
        quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16),
                         border=(5, 5), border_weight=2.0)  # Invalid!
    except ValueError as e:
        print(f"✓ Caught error: {e}")

    # Invalid border dimensions
    try:
        quilt = NCYXQuilt(Y=128, X=128, window=(32, 32), step=(16, 16),
                         border=(1, 2, 3))  # Wrong size for 2D!
    except ValueError as e:
        print(f"✓ Caught error: {e}")
