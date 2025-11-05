"""
Base classes and utilities for quilt operations.

This module provides base classes and shared utilities to eliminate code duplication
across 2D/3D and in-memory/disk-cached quilt implementations.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch


def normalize_border(
    border: Optional[Union[int, Tuple[int, ...]]], ndim: int
) -> Optional[Tuple[int, ...]]:
    """
    Normalize border parameter to a consistent format.

    Parameters
    ----------
    border : int, tuple, or None
        Border specification. Can be:
        - None: No border
        - int: Border size for all dimensions
        - tuple: Border size for each dimension
    ndim : int
        Number of dimensions (2 for 2D, 3 for 3D)

    Returns
    -------
    Optional[Tuple[int, ...]]
        Normalized border tuple or None if no border
    """
    if border is None:
        return None

    # Convert int to tuple
    if isinstance(border, int):
        if border == 0:
            return None
        return tuple([border] * ndim)

    # Handle tuple
    if isinstance(border, tuple):
        # Check if all zeros
        if all(b == 0 for b in border):
            return None
        # Ensure correct length
        if len(border) != ndim:
            raise ValueError(
                f"border tuple must have {ndim} elements for {ndim}D data, "
                f"got {len(border)}"
            )
        return border

    raise TypeError(f"border must be int, tuple, or None, got {type(border)}")


def validate_border_weight(border_weight: float) -> float:
    """
    Validate and normalize border_weight.

    Parameters
    ----------
    border_weight : float
        Weight for border pixels (0.0 to 1.0)

    Returns
    -------
    float
        Validated border_weight (clamped to [1e-8, 1.0])

    Raises
    ------
    ValueError
        If border_weight is outside valid range
    """
    if not (0.0 <= border_weight <= 1.0):
        raise ValueError(f"border_weight must be in [0.0, 1.0], got {border_weight}")
    return max(border_weight, 1e-8)


def compute_weight_matrix_torch(
    window: Tuple[int, ...], border: Optional[Tuple[int, ...]], border_weight: float
) -> torch.Tensor:
    """
    Compute weight matrix for stitching (torch version).

    Parameters
    ----------
    window : Tuple[int, ...]
        Window size for each dimension
    border : Optional[Tuple[int, ...]]
        Border size for each dimension, or None
    border_weight : float
        Weight for border pixels

    Returns
    -------
    torch.Tensor
        Weight matrix with same shape as window
    """
    weight = torch.ones(window)
    if border is not None:
        weight = torch.zeros(window) + border_weight
        # Set center region to 1.0
        slices = []
        for b in border:
            if b > 0:
                slices.append(slice(b, -b))
            else:
                slices.append(slice(None))
        weight[tuple(slices)] = 1.0
    return weight


def compute_border_tensor_torch(
    window: Tuple[int, ...], border: Optional[Tuple[int, ...]]
) -> torch.Tensor:
    """
    Compute border tensor (torch version).

    Returns 1.0 for valid regions (non-border), 0.0 for border regions.

    Parameters
    ----------
    window : Tuple[int, ...]
        Window size for each dimension
    border : Optional[Tuple[int, ...]]
        Border size for each dimension, or None

    Returns
    -------
    torch.Tensor
        Border tensor with same shape as window
    """
    if border is not None:
        result = torch.zeros(window)
        slices = []
        for b in border:
            if b > 0:
                slices.append(slice(b, -b))
            else:
                slices.append(slice(None))
        result[tuple(slices)] = 1.0
        return result
    else:
        return torch.ones(window)


def compute_weight_matrix_numpy(
    window: Tuple[int, ...], border: Optional[Tuple[int, ...]], border_weight: float
) -> np.ndarray:
    """
    Compute weight matrix for stitching (numpy version).

    Parameters
    ----------
    window : Tuple[int, ...]
        Window size for each dimension
    border : Optional[Tuple[int, ...]]
        Border size for each dimension, or None
    border_weight : float
        Weight for border pixels

    Returns
    -------
    np.ndarray
        Weight matrix with same shape as window
    """
    weight = np.ones(window, dtype=np.float64) * border_weight
    if border is not None:
        slices = []
        for b in border:
            if b > 0:
                slices.append(slice(b, -b))
            else:
                slices.append(slice(None))
        weight[tuple(slices)] = 1.0
    return weight


def compute_border_tensor_numpy(
    window: Tuple[int, ...], border: Optional[Tuple[int, ...]]
) -> np.ndarray:
    """
    Compute border tensor (numpy version).

    Returns 1.0 for valid regions (non-border), 0.0 for border regions.

    Parameters
    ----------
    window : Tuple[int, ...]
        Window size for each dimension
    border : Optional[Tuple[int, ...]]
        Border size for each dimension, or None

    Returns
    -------
    np.ndarray
        Border tensor with same shape as window
    """
    result = np.ones(window, dtype=np.float64)
    if border is not None:
        result = result - 1  # Set to zeros
        slices = []
        for b in border:
            if b > 0:
                slices.append(slice(b, -b))
            else:
                slices.append(slice(None))
        result[tuple(slices)] = 1.0
    return result


def compute_chunk_times(
    dimension_sizes: Tuple[int, ...], window: Tuple[int, ...], step: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Compute number of chunks along each dimension.

    Ensures the last chunk is included by adjusting starting points.

    Parameters
    ----------
    dimension_sizes : Tuple[int, ...]
        Size of each dimension
    window : Tuple[int, ...]
        Window size for each dimension
    step : Tuple[int, ...]
        Step size for each dimension

    Returns
    -------
    Tuple[int, ...]
        Number of chunks along each dimension
    """

    def compute_steps(dimension_size: int, window_size: int, step_size: int) -> int:
        """Calculate number of steps needed."""
        full_steps = (dimension_size - window_size) // step_size
        # Check if there is enough space left for the last chunk
        if dimension_size > full_steps * step_size + window_size:
            return full_steps + 2
        else:
            return full_steps + 1

    return tuple(
        compute_steps(dim_size, win_size, step_size)
        for dim_size, win_size, step_size in zip(dimension_sizes, window, step)
    )


class BaseQuilt:
    """
    Base class for all quilt operations.

    Provides common initialization and validation logic.
    """

    def __init__(
        self,
        window: Tuple[int, ...],
        step: Tuple[int, ...],
        border: Optional[Union[int, Tuple[int, ...]]],
        border_weight: float,
        ndim: int,
    ) -> None:
        """
        Initialize base quilt.

        Parameters
        ----------
        window : Tuple[int, ...]
            Window size for each dimension
        step : Tuple[int, ...]
            Step size for each dimension
        border : Optional[Union[int, Tuple[int, ...]]]
            Border specification
        border_weight : float
            Weight for border pixels
        ndim : int
            Number of dimensions (2 or 3)
        """
        # Validate and normalize inputs
        self.border = normalize_border(border, ndim)
        self.border_weight = validate_border_weight(border_weight)
        self.window = window
        self.step = step
        self.ndim = ndim

        # Validate window and step match dimensions
        if len(window) != ndim:
            raise ValueError(
                f"window must have {ndim} elements for {ndim}D data, got {len(window)}"
            )
        if len(step) != ndim:
            raise ValueError(
                f"step must have {ndim} elements for {ndim}D data, got {len(step)}"
            )

        # Validate border matches dimensions if provided
        if self.border is not None and len(self.border) != ndim:
            raise ValueError(
                f"border must have {ndim} elements for {ndim}D data, "
                f"got {len(self.border)}"
            )
