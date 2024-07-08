# qlty

qlty is a Python library designed to handle large 2D or 3D tensors efficiently by splitting them into smaller, manageable chunks. This library is particularly useful for processing large datasets that do not fit into memory. It includes methods for stitching the smaller chunks back together and computing the mean and standard deviation of the reconstructed data.

## Modules and Features
- **qlty2D** : in-memory functionality, requires things to fit into CPU memory. 

- **qlty2DLarge**: Similar functionality as qlty2D, but uses on-disc caching. If you ever think you want to graduate to the 3D case, this is what to use.

- **qlty3Dlarge**: Use this for 3D datasets, has on-disc caching of intermediate results.

Tasks:
- **unstich**: Split large tensors into smaller chunks for efficient processing.
- **stitch**: Reassemble the smaller chunks into the original tensor, with potentially overlaps taken into account
- **Border Handling**: Manage border pixels to minimize artifacts during stitching and training in segmentation.

## Installation

To install the required dependencies, you can use `pip`:

```bash
pip install qlty torch zarr numpy einops dask


