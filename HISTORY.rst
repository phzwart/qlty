=======
History
=======

1.3.0 (2025-01-XX)
------------------

* Added ``pretokenizer_2d`` module for pre-tokenization of patches with sequence model support (2D).
* New functions: ``tokenize_patch()`` and ``build_sequence_pair()`` for converting patches
  into token sequences with overlap information.
* Optimized batch processing with numba JIT compilation and parallel execution.
* Supports both single patches and batched processing for efficient training workflows.
* Designed for self-supervised learning, contrastive learning, and transformer-based models.

1.2.3 (2025-11-23)
------------------

* **New: 2.5D Quilt Module** - Added ``NCZYX25DQuilt`` class for converting 3D volumetric data
  (N, C, Z, Y, X) into 2.5D multi-channel data by slicing the Z dimension into channels.
  Supports flexible channel specifications (identity, mean, std operations), selective z-slice
  processing, and two accumulation modes (2D planes or 3D stack).
* **New: Backend System** - Added comprehensive backend support for various data sources:
  * ``InMemoryBackend``: Wraps torch.Tensor for in-memory data
  * ``ZarrBackend``: On-demand loading from OME-Zarr files
  * ``HDF5Backend``: On-demand loading from HDF5 datasets
  * ``MemoryMappedBackend``: Memory-mapped numpy arrays
  * ``TensorLike3D``: Unified tensor-like interface for all backends
  * Convenience functions: ``from_zarr()``, ``from_hdf5()``, ``from_memmap()``
* **New: Image Stack to Zarr Utility** - Added ``stack_files_to_zarr()`` function in
  ``qlty.utils.stack_to_zarr`` for converting image file stacks (TIFF, PNG, etc.) into
  efficient Zarr format with automatic pattern matching, gap detection, and metadata storage.
* **New: False Color Visualization** - Added ``FalseColorGenerator`` class in ``qlty.utils.false_color_2D``
  for creating UMAP-based false-color visualizations of 2D images using patch-based dimensionality
  reduction.
* **Improved Test Coverage** - Added 65+ new tests across qlty2_5D, backends_2_5D, and stack_to_zarr
  modules, significantly improving coverage:
  * ``qlty2_5D.py``: 75% → 88% coverage
  * ``backends_2_5D.py``: 62% → 70% coverage
  * ``stack_to_zarr.py``: 38% → 94% coverage
* **CI Improvements** - Fixed coverage reporting in CI by using ``coverage run`` directly instead
  of pytest-cov to avoid torch import conflicts. Added coverage verification steps.

1.2.0 (2025-11-13)
------------------

* Added optional rotation-aware extraction for 2D patch pairs with matching overlap handling.
* Expanded tests and documentation to cover rotated patch workflows.

1.1.0 (2025-11-12)
------------------

* Restored Numba acceleration for 2D quilting via color-based parallel stitching that avoids write races.
* Expanded 3D patch-pair sampling tests to cover edge cases and fallback logic, driving coverage to 100%.
* Updated documentation to describe partially overlapping patch-pair utilities.
* Noted that NCZYXQuilt and the Large* variants still need analogous race-free acceleration.

0.1.0 (2021-10-20)
------------------

* First release on PyPI.

0.1.1 (some time ago)
---------------------

* Minor bug fixes

0.1.2. (2022-9-13)
------------------

* Support for N-channel 3D tensors
* On disc-accmulation for large datasets


0.1.3. (2022-9-13)
------------------

* Cleanup and ready to redistribute


0.1.4. (2023-8-28)
------------------

* Bug fix / behavoir change

0.1.5. (2023-12-28)
-------------------

* Changes to qlty3DLarge:
  bug fixes
  normalizing / averaging now done by dask

0.1.6. (2024-03-10)
-------------------
  bug fixes

0.1.7. (2024-03-12)
-------------------
*  bug fix in border tensor definition.

0.2.0. (2024-03-12)
-------------------
*  bug fixes
*  2DLarge, mimics 3DLarge
