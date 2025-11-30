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

Example 11: Pre-Tokenization for Patch Processing (2D)
--------------------------------------------------------

**What and Why**: The ``pretokenizer_2d`` module prepares patches for tokenization by
enabling sequence-based models (like transformers) to work with image patches. This is
useful for:

- **Self-supervised learning**: Learning representations from patch pairs with known
  geometric relationships
- **Contrastive learning**: Using overlapping tokens as positive pairs
- **Sequence models**: Converting 2D patches into token sequences with spatial awareness
- **Efficient batch processing**: Processing many patch pairs in parallel with numba
  acceleration

The key innovation is that it identifies which tokens overlap between two patches
that have undergone a known rigid transformation (translation + rotation), providing
the overlap information needed for training sequence-based models.

**Basic Usage - Single Patch Pair**::

    from qlty import extract_patch_pairs, build_sequence_pair, tokenize_patch
    import torch

    # Step 1: Extract patch pairs using qlty's existing functionality
    images = torch.randn(5, 3, 128, 128)
    patches1, patches2, deltas, rotations = extract_patch_pairs(
        images,
        window=(64, 64),
        num_patches=10,
        delta_range=(10.0, 20.0),
        random_seed=42
    )

    # Step 2: Build sequence pairs with overlap information
    # This tokenizes both patches and finds overlapping tokens
    result = build_sequence_pair(
        patches1[0],           # First patch: (3, 64, 64)
        patches2[0],           # Second patch: (3, 64, 64)
        dx=deltas[0, 0].item(),  # Translation in x
        dy=deltas[0, 1].item(),  # Translation in y
        rot_k90=rotations[0].item(),  # Rotation (0, 1, 2, or 3 for 0°, 90°, 180°, 270°)
        patch_size=16,         # Size of each token
        stride=8               # Stride for overlapping tokens (default: patch_size//2)
    )

    # Result contains:
    print(f"Tokens from patch1: {result['tokens1'].shape}")  # (T, D) where T=number of tokens
    print(f"Tokens from patch2: {result['tokens2'].shape}")  # (T, D)
    print(f"Overlapping tokens: {result['overlap_mask1'].sum().item()} out of {result['tokens1'].shape[0]}")

    # Use for training:
    # - tokens1, tokens2: Input to your sequence model (e.g., transformer)
    # - coords1, coords2: Absolute coordinates for positional encoding
    # - overlap_mask1, overlap_mask2: Which tokens have corresponding overlaps
    # - overlap_indices1_to_2: Mapping from patch1 tokens to patch2 tokens
    # - overlap_fractions: How much each token overlaps (0.0 to 1.0)

**Batch Processing - Efficient for Large Datasets**::

    # Process all patch pairs at once (much faster!)
    batch_result = build_sequence_pair(
        patches1,              # (50, 3, 64, 64) - batch of patches
        patches2,              # (50, 3, 64, 64)
        dx=deltas[:, 0],       # (50,) - x translations
        dy=deltas[:, 1],       # (50,) - y translations
        rot_k90=rotations,     # (50,) - rotations
        patch_size=16,
        stride=8
    )

    # Batch result has padded tensors for efficient processing
    print(f"Batch tokens1: {batch_result['tokens1'].shape}")  # (50, T_max, D)
    print(f"Sequence lengths: {batch_result['sequence_lengths']}")  # (50,) - actual lengths
    print(f"Overlap counts: {batch_result['overlap_pair_counts']}")  # (50,) - overlaps per pair

    # Use sequence_lengths to mask padding in your model
    # Use overlap_pair_counts to understand data distribution

**Tokenization Only - When You Just Need Tokens**::

    # If you only need to tokenize a patch (no overlap computation)
    patch = torch.randn(3, 64, 64)
    tokens, coords = tokenize_patch(patch, patch_size=16, stride=8)

    print(f"Created {tokens.shape[0]} tokens")
    print(f"Token shape: {tokens.shape[1]}")  # 3*16*16 = 768 dimensions
    print(f"Coordinates shape: {coords.shape}")  # (T, 2) - (y, x) for each token

    # Use tokens as input to sequence models
    # Use coords for positional encoding

**Real-World Use Case - Self-Supervised Learning**::

    from qlty import extract_patch_pairs, build_sequence_pair
    import torch
    import torch.nn as nn

    # Extract patch pairs from unlabeled images
    images = torch.randn(100, 3, 256, 256)
    patches1, patches2, deltas, rotations = extract_patch_pairs(
        images, window=(128, 128), num_patches=20, delta_range=(20.0, 40.0)
    )

    # Build sequence pairs
    batch_result = build_sequence_pair(
        patches1, patches2, deltas[:, 0], deltas[:, 1], rotations,
        patch_size=32, stride=16
    )

    # Train a transformer to predict overlapping tokens
    class PatchTransformer(nn.Module):
        def __init__(self, token_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Linear(token_dim, hidden_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_dim, nhead=8), num_layers=6
            )
            self.predictor = nn.Linear(hidden_dim, token_dim)

        def forward(self, tokens, coords, mask):
            # Add positional encoding from coords
            pos_enc = self.positional_encoding(coords)
            x = self.embedding(tokens) + pos_enc
            x = self.transformer(x)
            return self.predictor(x)

    model = PatchTransformer(token_dim=3*32*32, hidden_dim=512)

    # Training loop
    for epoch in range(10):
        for i in range(0, len(patches1), 32):  # Process in batches
            batch_idx = slice(i, i+32)
            result = build_sequence_pair(
                patches1[batch_idx], patches2[batch_idx],
                deltas[batch_idx, 0], deltas[batch_idx, 1], rotations[batch_idx],
                patch_size=32, stride=16
            )

            # Get overlapping tokens
            tokens1 = result['tokens1']  # (32, T_max, D)
            tokens2 = result['tokens2']  # (32, T_max, D)
            overlap_mask = result['overlap_mask1']  # (32, T_max)
            overlap_indices = result['overlap_indices1_to_2']  # (32, T_max)

            # Predict tokens2 from tokens1
            predicted = model(tokens1, result['coords1'], overlap_mask)

            # Loss only on overlapping tokens
            # (simplified - actual implementation would handle padding)
            loss = nn.functional.mse_loss(
                predicted[overlap_mask],
                tokens2[overlap_mask]
            )

            # Backprop and update...

**Performance Notes**:

- **Batch processing is highly optimized**: Uses numba JIT compilation and parallel
  processing for large batches (N > 5)
- **Automatic fallback**: Falls back to sequential processing for small batches or
  when numba is unavailable
- **Memory efficient**: Batch tokenization reuses a single ``NCYXQuilt`` object
- **GPU support**: All tensors maintain device placement (CPU/GPU)

**When to Use**:

- ✅ Training sequence models (transformers) on image patches
- ✅ Self-supervised learning with geometric augmentations
- ✅ Contrastive learning with patch pairs
- ✅ Any task requiring token-level overlap information

**When NOT to Use**:

- ❌ Simple patch extraction (use ``NCYXQuilt.unstitch()`` instead)
- ❌ Stitching patches back together (use ``NCYXQuilt.stitch()`` instead)
- ❌ When you don't need overlap information

Example 12: Metadata Extraction and Zarr Storage
--------------------------------------------------

**Use Case**: Extract patch pair metadata for a large dataset, then selectively load patches or save to Zarr for efficient storage.

**Step 1: Extract Metadata Only**::

    from qlty.patch_pairs_2d import extract_patch_pairs_metadata
    import torch

    # Large dataset: 1000 images
    tensor = torch.randn(1000, 3, 256, 256)
    window = (64, 64)
    num_patches = 100  # 100 patch pairs per image = 100,000 total

    # Extract metadata (fast, doesn't load patches)
    metadata = extract_patch_pairs_metadata(
        tensor, window, num_patches, delta_range=(10.0, 20.0),
        random_seed=42,
        num_workers=8  # Parallel processing
    )

    print(f"Extracted metadata for {len(metadata['patch1_x'])} patch pairs")

**Step 2: Save to Zarr Format**::

    from qlty.patch_pairs_2d import extract_patches_to_zarr
    import zarr

    # Save all patches to Zarr
    zarr_path = "large_patches.zarr"
    group = zarr.open_group(zarr_path, mode="w")

    extract_patches_to_zarr(
        tensor, metadata, group,
        chunk_size=(1000, 3, 64, 64)  # Optimize for batch loading
    )

    print("Patches saved to Zarr format")

**Step 3: Load with PyTorch DataLoader**::

    from qlty.patch_pairs_2d import ZarrPatchPairDataset
    from torch.utils.data import DataLoader

    # Open Zarr group
    group = zarr.open_group("large_patches.zarr", mode="r")
    dataset = ZarrPatchPairDataset(group)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    # Train model
    for epoch in range(10):
        for patches1, patches2, deltas, rotations in dataloader:
            # Training code...
            output1 = model(patches1)
            output2 = model(patches2)
            loss = compute_loss(output1, output2, deltas)
            loss.backward()

**Benefits**:
- **Memory efficient**: Metadata extraction is fast and doesn't load patches
- **Selective loading**: Extract only patches you need
- **Zarr storage**: Efficient chunked storage for large datasets
- **DataLoader integration**: Seamless PyTorch integration

Example 13: Converting Image Stacks to OME-Zarr
-------------------------------------------------

**Use Case**: Convert a directory of TIFF images into OME-Zarr format with multiscale pyramids for efficient viewing and processing.

**Basic Conversion**::

    from qlty.utils.stack_to_zarr import stack_files_to_ome_zarr
    from pathlib import Path

    # Directory structure:
    # images/
    #   stack1_001.tif
    #   stack1_002.tif
    #   ...
    #   stack1_100.tif
    #   stack2_001.tif
    #   ...

    result = stack_files_to_ome_zarr(
        directory="images",
        extension=".tif",
        pattern=r"(.+)_(\d+)\.tif$",  # Matches basename and counter
        axis_order="ZCYX",  # Z (depth), Channel, Y, X
        pyramid_levels=4,
        downsample_mode="2d",
        downsample_method="dask"
    )

    # result contains metadata for each stack:
    # {
    #     "stack1": {
    #         "zarr_path": "images/stack1.zarr",
    #         "shape": (100, 3, 512, 512),
    #         "file_count": 100,
    #         "pyramid_levels": 4
    #     },
    #     "stack2": {...}
    # }

**Viewing with napari**::

    import napari
    import zarr

    # Open OME-Zarr file
    zarr_path = result["stack1"]["zarr_path"]
    group = zarr.open_group(zarr_path, mode="r")

    # Load full resolution
    data = group["0"][:]  # Shape: (Z, C, Y, X)

    # View in napari (supports multiscale automatically)
    viewer = napari.Viewer()
    viewer.add_image(data, multiscale=True)
    napari.run()

**Benefits**:
- **Multiscale pyramids**: Efficient viewing at different zoom levels
- **Standard format**: Compatible with bioimaging tools
- **Chunked storage**: Fast random access
- **Rich metadata**: OME metadata stored automatically
