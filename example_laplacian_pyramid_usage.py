"""
Simple example: Using Laplacian Pyramid OME-Zarr

Copy and paste this into your notebook or script.
"""

from pathlib import Path

import numpy as np
import tifffile

from qlty.utils.stack_to_zarr import (
    stack_files_to_ome_zarr_laplacian,
)

# Create test images
temp_dir = Path("temp_images")
temp_dir.mkdir(exist_ok=True)

for i in range(5):
    img = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    tifffile.imwrite(temp_dir / f"test_{i:02d}.tif", img)

# Create Laplacian pyramid
result = stack_files_to_ome_zarr_laplacian(
    directory=temp_dir,
    extension=".tif",
    pattern=r"(.+)_(\d+)\.tif$",
    pyramid_levels=3,
    interpolation_mode="bilinear",
    store_base_level=True,
    verbose=True,
)

print(f"Created Laplacian pyramid: {result}")

