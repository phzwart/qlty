"""
Backend implementations for 2.5D Quilt data sources.

Provides tensor-like interface for various data storage formats:
- In-memory torch.Tensor
- Memory-mapped tensors
- OME-Zarr files
- HDF5 files
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


class DataSource3DBackend(ABC):
    """
    Backend interface for actual data storage/loading.
    Implementations handle the specifics of different data sources.
    """

    @abstractmethod
    def get_shape(self) -> Tuple[int, int, int, int, int]:
        """
        Return (N, C, Z, Y, X) shape.

        Returns
        -------
        Tuple[int, int, int, int, int]
            Shape as (N, C, Z, Y, X)
        """
        pass

    @abstractmethod
    def get_dtype(self) -> torch.dtype:
        """
        Return data type (as torch.dtype).

        Returns
        -------
        torch.dtype
            Data type of the source
        """
        pass

    @abstractmethod
    def load_slice(
        self,
        n: Optional[int] = None,
        c: Optional[int] = None,
        z: Optional[Union[int, slice]] = None,
        y: Optional[Union[int, slice]] = None,
        x: Optional[Union[int, slice]] = None,
    ) -> torch.Tensor:
        """
        Load data slice and return as torch.Tensor.
        Loads only what's requested - never entire dataset.

        Parameters
        ----------
        n, c, z, y, x : int, slice, or None
            Indices/slices for each dimension. None means all.

        Returns
        -------
        torch.Tensor
            Requested slice as PyTorch tensor
        """
        pass

    @property
    @abstractmethod
    def supports_batch_loading(self) -> bool:
        """
        Whether backend can efficiently load multiple z-slices at once.

        Returns
        -------
        bool
            True if batch loading is supported
        """
        pass

    def get_z_slices(self, n: int, c: int, z_indices: List[int]) -> torch.Tensor:
        """
        Optional: Batch loading of multiple z-slices.

        Parameters
        ----------
        n : int
            Image index
        c : int
            Channel index
        z_indices : List[int]
            List of z-slice indices to load

        Returns
        -------
        torch.Tensor
            Stacked z-slices of shape (len(z_indices), Y, X)
        """
        if self.supports_batch_loading:
            # Optimized batch implementation
            z_min = min(z_indices)
            z_max = max(z_indices) + 1
            return self.load_slice(n=n, c=c, z=slice(z_min, z_max))
        else:
            # Fallback to individual calls
            slices = [self.load_slice(n=n, c=c, z=z) for z in z_indices]
            return torch.stack(slices, dim=0)


class InMemoryBackend(DataSource3DBackend):
    """
    Backend for in-memory torch.Tensor.
    No-op wrapper that just returns views.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of shape (N, C, Z, Y, X)
        """
        if len(tensor.shape) != 5:
            raise ValueError(
                f"Tensor must be 5D (N, C, Z, Y, X), got shape {tensor.shape}"
            )
        self.tensor = tensor

    def get_shape(self) -> Tuple[int, int, int, int, int]:
        return tuple(self.tensor.shape)

    def get_dtype(self) -> torch.dtype:
        return self.tensor.dtype

    def load_slice(
        self,
        n: Optional[int] = None,
        c: Optional[int] = None,
        z: Optional[Union[int, slice]] = None,
        y: Optional[Union[int, slice]] = None,
        x: Optional[Union[int, slice]] = None,
    ) -> torch.Tensor:
        # Build indexing tuple
        indices = [slice(None)] * 5
        if n is not None:
            indices[0] = n
        if c is not None:
            indices[1] = c
        if z is not None:
            indices[2] = z
        if y is not None:
            indices[3] = y
        if x is not None:
            indices[4] = x

        return self.tensor[tuple(indices)]

    @property
    def supports_batch_loading(self) -> bool:
        return True


class MemoryMappedBackend(DataSource3DBackend):
    """
    Backend for memory-mapped numpy arrays.
    Loads data on-demand from memory-mapped file.
    """

    def __init__(self, mmap_array: np.memmap, dtype: Optional[torch.dtype] = None):
        """
        Parameters
        ----------
        mmap_array : np.memmap
            Memory-mapped numpy array of shape (N, C, Z, Y, X)
        dtype : Optional[torch.dtype]
            Target dtype for conversion. If None, uses array's dtype.
        """
        if len(mmap_array.shape) != 5:
            raise ValueError(
                f"Array must be 5D (N, C, Z, Y, X), got shape {mmap_array.shape}"
            )
        self.mmap_array = mmap_array
        self._dtype = (
            dtype or torch.from_numpy(np.array([], dtype=mmap_array.dtype)).dtype
        )

    def get_shape(self) -> Tuple[int, int, int, int, int]:
        return tuple(self.mmap_array.shape)

    def get_dtype(self) -> torch.dtype:
        return self._dtype

    def load_slice(
        self,
        n: Optional[int] = None,
        c: Optional[int] = None,
        z: Optional[Union[int, slice]] = None,
        y: Optional[Union[int, slice]] = None,
        x: Optional[Union[int, slice]] = None,
    ) -> torch.Tensor:
        # Build indexing tuple
        indices = [slice(None)] * 5
        if n is not None:
            indices[0] = n
        if c is not None:
            indices[1] = c
        if z is not None:
            indices[2] = z
        if y is not None:
            indices[3] = y
        if x is not None:
            indices[4] = x

        # Load from memory-mapped array
        data = self.mmap_array[tuple(indices)]
        # Copy to avoid non-writable tensor warning
        if not data.flags.writeable:
            data = data.copy()
        return torch.from_numpy(data).to(self._dtype)

    @property
    def supports_batch_loading(self) -> bool:
        return True


class ZarrBackend(DataSource3DBackend):
    """
    Backend for OME-Zarr files.
    Loads data on-demand from zarr arrays.
    """

    def __init__(self, zarr_array, dtype: Optional[torch.dtype] = None):
        """
        Parameters
        ----------
        zarr_array : zarr.Array
            Zarr array of shape (N, C, Z, Y, X) or (C, Z, Y, X) or (Z, Y, X)
        dtype : Optional[torch.dtype]
            Target dtype for conversion. If None, infers from zarr array.
        """
        try:
            import zarr  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "zarr is required for ZarrBackend. Install with: pip install zarr"
            ) from err

        self.zarr_array = zarr_array

        # Handle different zarr array shapes
        shape = zarr_array.shape
        if len(shape) == 3:
            # (Z, Y, X) - single image, single channel
            self.shape = (1, 1) + shape
        elif len(shape) == 4:
            # (C, Z, Y, X) - single image, multiple channels
            self.shape = (1,) + shape
        elif len(shape) == 5:
            # (N, C, Z, Y, X) - multiple images
            self.shape = shape
        else:
            raise ValueError(f"Zarr array must be 3D, 4D, or 5D, got shape {shape}")

        # Infer dtype
        if dtype is None:
            np_dtype = zarr_array.dtype
            self._dtype = torch.from_numpy(np.array([], dtype=np_dtype)).dtype
        else:
            self._dtype = dtype

    def get_shape(self) -> Tuple[int, int, int, int, int]:
        return self.shape

    def get_dtype(self) -> torch.dtype:
        return self._dtype

    def load_slice(
        self,
        n: Optional[int] = None,
        c: Optional[int] = None,
        z: Optional[Union[int, slice]] = None,
        y: Optional[Union[int, slice]] = None,
        x: Optional[Union[int, slice]] = None,
    ) -> torch.Tensor:
        # Build indexing tuple based on array dimensionality
        original_shape = self.zarr_array.shape
        if len(original_shape) == 3:
            # (Z, Y, X) - treat as single image, single channel
            # n and c are ignored (always n=0, c=0)
            indices = [slice(None)] * 3
            if z is not None:
                indices[0] = z
            if y is not None:
                indices[1] = y
            if x is not None:
                indices[2] = x
        elif len(original_shape) == 4:
            # (C, Z, Y, X) - treat as single image, multiple channels
            # n is ignored (always n=0)
            indices = [slice(None)] * 4
            if c is not None:
                indices[0] = c
            if z is not None:
                indices[1] = z
            if y is not None:
                indices[2] = y
            if x is not None:
                indices[3] = x
        else:
            # (N, C, Z, Y, X) - full 5D array
            indices = [slice(None)] * 5
            if n is not None:
                indices[0] = n
            if c is not None:
                indices[1] = c
            if z is not None:
                indices[2] = z
            if y is not None:
                indices[3] = y
            if x is not None:
                indices[4] = x

        # Load from zarr
        data = self.zarr_array[tuple(indices)]

        # Convert to torch tensor
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(self._dtype)
        else:
            # zarr might return a different type
            tensor = torch.from_numpy(np.array(data)).to(self._dtype)

        # Normalize to expected output dimensions based on what was requested
        # If n is an int, return (C, Z, Y, X) - no N dimension
        # If n is None or slice, return (N, C, Z, Y, X) - full 5D
        n_is_int = isinstance(n, int)
        original_shape = self.zarr_array.shape

        if len(original_shape) == 3:
            # Original is (Z, Y, X)
            if n_is_int:
                # Requested specific n (integer): return (C, Z, Y, X) = (1, Z, Y, X)
                if tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0)  # Add C dimension: (1, Z, Y, X)
                elif tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, Y, X)
            else:
                # No n or slice: return (N, C, Z, Y, X) = (1, 1, Z, Y, X)
                if tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
                elif tensor.ndim == 2:
                    tensor = (
                        tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    )  # (1, 1, 1, Y, X)
        elif len(original_shape) == 4:
            # Original is (C, Z, Y, X)
            if n_is_int:
                # Requested specific n (integer): return (C, Z, Y, X) - already correct
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0).unsqueeze(
                        0
                    )  # (1, 1, Y, X) if single slice
            else:
                # No n or slice: return (N, C, Z, Y, X) = (1, C, Z, Y, X)
                if tensor.ndim == 4:
                    tensor = tensor.unsqueeze(0)  # Add N dimension: (1, C, Z, Y, X)
                elif tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, ...)

        return tensor

    @property
    def supports_batch_loading(self) -> bool:
        return True


class HDF5Backend(DataSource3DBackend):
    """
    Backend for HDF5 files.
    Loads data on-demand from HDF5 datasets.
    """

    def __init__(self, h5_dataset, dtype: Optional[torch.dtype] = None):
        """
        Parameters
        ----------
        h5_dataset : h5py.Dataset
            HDF5 dataset of shape (N, C, Z, Y, X) or compatible
        dtype : Optional[torch.dtype]
            Target dtype for conversion. If None, infers from dataset.
        """
        try:
            import h5py  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "h5py is required for HDF5Backend. Install with: pip install h5py"
            ) from err

        self.h5_dataset = h5_dataset

        # Handle different dataset shapes
        shape = h5_dataset.shape
        if len(shape) == 3:
            # (Z, Y, X) - single image, single channel
            self.shape = (1, 1) + shape
        elif len(shape) == 4:
            # (C, Z, Y, X) - single image, multiple channels
            self.shape = (1,) + shape
        elif len(shape) == 5:
            # (N, C, Z, Y, X) - multiple images
            self.shape = shape
        else:
            raise ValueError(f"HDF5 dataset must be 3D, 4D, or 5D, got shape {shape}")

        # Infer dtype
        if dtype is None:
            np_dtype = h5_dataset.dtype
            self._dtype = torch.from_numpy(np.array([], dtype=np_dtype)).dtype
        else:
            self._dtype = dtype

    def get_shape(self) -> Tuple[int, int, int, int, int]:
        return self.shape

    def get_dtype(self) -> torch.dtype:
        return self._dtype

    def load_slice(
        self,
        n: Optional[int] = None,
        c: Optional[int] = None,
        z: Optional[Union[int, slice]] = None,
        y: Optional[Union[int, slice]] = None,
        x: Optional[Union[int, slice]] = None,
    ) -> torch.Tensor:
        # Build indexing tuple based on dataset dimensionality
        original_shape = self.h5_dataset.shape
        if len(original_shape) == 3:
            # (Z, Y, X)
            indices = [slice(None)] * 3
            if z is not None:
                indices[0] = z
            if y is not None:
                indices[1] = y
            if x is not None:
                indices[2] = x
        elif len(original_shape) == 4:
            # (C, Z, Y, X)
            indices = [slice(None)] * 4
            if c is not None:
                indices[0] = c
            if z is not None:
                indices[1] = z
            if y is not None:
                indices[2] = y
            if x is not None:
                indices[3] = x
        else:
            # (N, C, Z, Y, X)
            indices = [slice(None)] * 5
            if n is not None:
                indices[0] = n
            if c is not None:
                indices[1] = c
            if z is not None:
                indices[2] = z
            if y is not None:
                indices[3] = y
            if x is not None:
                indices[4] = x

        # Load from HDF5
        data = self.h5_dataset[tuple(indices)]

        # Convert to torch tensor
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self._dtype)
        else:
            return torch.from_numpy(np.array(data)).to(self._dtype)

    @property
    def supports_batch_loading(self) -> bool:
        return True


class TensorLike3D:
    """
    Wrapper that makes any data source look like a PyTorch tensor.
    Supports tensor-like indexing but loads data on-demand.
    """

    def __init__(self, backend: DataSource3DBackend):
        """
        Parameters
        ----------
        backend : DataSource3DBackend
            Backend that actually stores/loads the data
        """
        self.backend = backend
        self._shape = backend.get_shape()  # (N, C, Z, Y, X)
        self._dtype = backend.get_dtype()

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        """Return (N, C, Z, Y, X) shape - like tensor.shape"""
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype - like tensor.dtype"""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Return device - defaults to CPU for on-demand loading"""
        return torch.device("cpu")

    def __getitem__(self, key) -> torch.Tensor:
        """
        Tensor-like indexing that returns PyTorch tensors.
        Loads data on-demand from backend.

        Examples:
        - data[0] -> (C, Z, Y, X) tensor
        - data[0, 1] -> (Z, Y, X) tensor
        - data[0, 1, 5:10] -> (5, Y, X) tensor
        - data[0, 1, 5] -> (Y, X) tensor

        Parameters
        ----------
        key : int, slice, tuple
            Indexing key (supports standard numpy/torch indexing)

        Returns
        -------
        torch.Tensor
            Requested slice as PyTorch tensor
        """
        # Handle different key types
        if isinstance(key, int):
            # Single index: data[0] -> (C, Z, Y, X)
            return self.backend.load_slice(n=key)
        elif isinstance(key, slice):
            # Slice: data[0:2] -> (2, C, Z, Y, X)
            return self.backend.load_slice(n=key)
        elif isinstance(key, tuple):
            # Multiple indices: data[0, 1, 5:10] -> (5, Y, X)
            # Parse tuple
            n = key[0] if len(key) > 0 else None
            c = key[1] if len(key) > 1 else None
            z = key[2] if len(key) > 2 else None
            y = key[3] if len(key) > 3 else None
            x = key[4] if len(key) > 4 else None

            return self.backend.load_slice(n=n, c=c, z=z, y=y, x=x)
        else:
            raise TypeError(f"Unsupported indexing type: {type(key)}")

    def __len__(self) -> int:
        """Return number of images (N dimension)"""
        return self._shape[0]

    def __repr__(self) -> str:
        return f"TensorLike3D(shape={self.shape}, dtype={self.dtype}, backend={type(self.backend).__name__})"


# Convenience functions for creating TensorLike3D from different sources


def from_zarr(zarr_path: str, dtype: Optional[torch.dtype] = None) -> TensorLike3D:
    """
    Create TensorLike3D from zarr file.

    Parameters
    ----------
    zarr_path : str
        Path to zarr file or zarr group
    dtype : Optional[torch.dtype]
        Target dtype for conversion. If None, infers from zarr array.

    Returns
    -------
    TensorLike3D
        Tensor-like wrapper around zarr backend

    Examples
    --------
    >>> data = from_zarr("data.zarr")
    >>> quilt = NCZYX25DQuilt(data, channel_spec={'direct': [0]})
    """
    try:
        import zarr  # noqa: F401
    except ImportError as err:
        raise ImportError("zarr is required. Install with: pip install zarr") from err

    z = zarr.open(zarr_path, mode="r")
    backend = ZarrBackend(z, dtype=dtype)
    return TensorLike3D(backend)


def from_hdf5(
    hdf5_path: str, dataset_path: str, dtype: Optional[torch.dtype] = None
) -> TensorLike3D:
    """
    Create TensorLike3D from HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 file
    dataset_path : str
        Path to dataset within HDF5 file (e.g., '/data' or '/images/stack')
    dtype : Optional[torch.dtype]
        Target dtype for conversion. If None, infers from dataset.

    Returns
    -------
    TensorLike3D
        Tensor-like wrapper around HDF5 backend

    Examples
    --------
    >>> data = from_hdf5("data.h5", "/images/stack")
    >>> quilt = NCZYX25DQuilt(data, channel_spec={'direct': [0]})
    """
    try:
        import h5py  # noqa: F401
    except ImportError as err:
        raise ImportError("h5py is required. Install with: pip install h5py") from err

    f = h5py.File(hdf5_path, "r")
    backend = HDF5Backend(f[dataset_path], dtype=dtype)
    return TensorLike3D(backend)


def from_memmap(
    file_path: str, dtype: np.dtype, shape: Tuple[int, ...], mode: str = "r"
) -> TensorLike3D:
    """
    Create TensorLike3D from memory-mapped numpy array.

    Parameters
    ----------
    file_path : str
        Path to memory-mapped file
    dtype : np.dtype
        Data type of the array
    shape : Tuple[int, ...]
        Shape of the array (must be 5D: N, C, Z, Y, X)
    mode : str
        File mode: 'r' (read-only), 'r+' (read-write), 'c' (copy-on-write)

    Returns
    -------
    TensorLike3D
        Tensor-like wrapper around memory-mapped backend

    Examples
    --------
    >>> data = from_memmap("data.dat", dtype=np.float32, shape=(10, 3, 50, 200, 200))
    >>> quilt = NCZYX25DQuilt(data, channel_spec={'direct': [0]})
    """
    mmap = np.memmap(file_path, dtype=dtype, mode=mode, shape=shape)
    backend = MemoryMappedBackend(mmap)
    return TensorLike3D(backend)
