"""
2.5D Quilt Implementation

Converts 3D data (N, C, Z, Y, X) to 2.5D multi-channel data by slicing
the Z dimension into channels according to a flexible specification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class ZOperation(Enum):
    """Operations for extracting channels from Z dimension."""
    DIRECT = 1  # Single pixel extraction
    MEAN = 3   # Mean of multiple pixels
    # Future: MAX = 5, MIN = 6, MEDIAN = 7


@dataclass
class ChannelOperation:
    """Represents a single channel operation."""
    op_type: ZOperation
    offsets: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
    output_channels: int  # Number of channels this operation produces
    name: Optional[str] = None  # Optional name for this operation
    
    def get_required_z_range(self, z0: int) -> Tuple[int, int]:
        """
        Get [z_min, z_max] needed to compute this operation at z0.
        Useful for optimizing data loading.
        
        Parameters
        ----------
        z0 : int
            Center z-slice index
            
        Returns
        -------
        Tuple[int, int]
            (z_min, z_max) where z_max is exclusive (like Python slicing)
        """
        if self.op_type == ZOperation.DIRECT:
            # For DIRECT, offsets is a tuple of ints
            offsets = self.offsets if isinstance(self.offsets[0], int) else self.offsets[0]
            z_indices = [z0 + offset for offset in offsets]
            return (min(z_indices), max(z_indices) + 1)
        elif self.op_type == ZOperation.MEAN:
            # For MEAN, offsets is a tuple of tuples
            all_offsets = []
            for offset_group in self.offsets:
                if isinstance(offset_group, (list, tuple)):
                    all_offsets.extend(offset_group)
                else:
                    all_offsets.append(offset_group)
            z_indices = [z0 + offset for offset in all_offsets]
            return (min(z_indices), max(z_indices) + 1)
        return (z0, z0 + 1)


def parse_channel_spec(
    spec: Union[
        Dict[str, Union[List[int], List[List[int]]]],
        Dict[ZOperation, Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]],
    ]
) -> List[ChannelOperation]:
    """
    Parse channel specification into list of operations.
    Supports string keys (auto-converted to enum) or ZOperation enum keys.
    Normalizes to internal ZOperation enum representation.
    
    Parameters
    ----------
    spec : Dict
        Channel specification with string or ZOperation enum keys.
        String keys: {'direct': [-1,0,1], 'mean': [[-1,-2,-3], [1,2,3]]}
        Enum keys: {ZOperation.DIRECT: (-1,0,1), ZOperation.MEAN: ...}
        
    Returns
    -------
    List[ChannelOperation]
        Ordered list of channel operations to apply
        
    Raises
    ------
    ValueError
        If spec is empty, contains invalid keys, or has malformed values
    """
    if not spec:
        raise ValueError("Channel specification cannot be empty")
    
    operations = []
    
    # String to enum mapping
    string_to_enum = {
        'direct': ZOperation.DIRECT,
        'mean': ZOperation.MEAN,
    }
    
    for key, value in spec.items():
        # Convert string keys to enum
        if isinstance(key, str):
            key_lower = key.lower()
            if key_lower not in string_to_enum:
                raise ValueError(
                    f"Unknown operation '{key}'. Supported operations: {list(string_to_enum.keys())}"
                )
            op_type = string_to_enum[key_lower]
        elif isinstance(key, ZOperation):
            op_type = key
        else:
            raise ValueError(
                f"Invalid key type: {type(key)}. Must be str or ZOperation enum"
            )
        
        # Process value based on operation type
        if op_type == ZOperation.DIRECT:
            # DIRECT: value should be a list/tuple of ints
            if isinstance(value, (list, tuple)):
                offsets = tuple(int(x) for x in value)
            else:
                raise ValueError(f"DIRECT operation requires list/tuple of ints, got {type(value)}")
            
            if not offsets:
                raise ValueError("DIRECT operation must have at least one offset")
            
            output_channels = len(offsets)
            operations.append(ChannelOperation(
                op_type=op_type,
                offsets=offsets,
                output_channels=output_channels,
                name=key if isinstance(key, str) else None
            ))
            
        elif op_type == ZOperation.MEAN:
            # MEAN: value should be a list/tuple of lists/tuples of ints
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"MEAN operation requires list/tuple of lists, got {type(value)}")
            
            if not value:
                raise ValueError("MEAN operation must have at least one offset group")
            
            # Normalize to tuple of tuples
            offset_groups = []
            for group in value:
                if isinstance(group, (list, tuple)):
                    group_tuple = tuple(int(x) for x in group)
                    if not group_tuple:
                        raise ValueError("MEAN operation offset groups cannot be empty")
                    offset_groups.append(group_tuple)
                else:
                    raise ValueError(f"MEAN operation offset groups must be lists/tuples, got {type(group)}")
            
            output_channels = len(offset_groups)
            operations.append(ChannelOperation(
                op_type=op_type,
                offsets=tuple(offset_groups),
                output_channels=output_channels,
                name=key if isinstance(key, str) else None
            ))
    
    return operations


def apply_boundary_mode(
    z_index: int,
    z_min: int,
    z_max: int,
    boundary_mode: str = "clamp"
) -> int:
    """
    Apply boundary mode to z_index to ensure it's within [z_min, z_max).
    
    Parameters
    ----------
    z_index : int
        Requested z index
    z_min : int
        Minimum valid z index (inclusive)
    z_max : int
        Maximum valid z index (exclusive)
    boundary_mode : str
        Boundary handling mode: "clamp", "zero", "reflect", "skip"
        
    Returns
    -------
    int
        Clamped/reflected z index, or -1 if skip mode and out of bounds
    """
    if z_min <= z_index < z_max:
        return z_index
    
    if boundary_mode == "clamp":
        return max(z_min, min(z_index, z_max - 1))
    elif boundary_mode == "zero":
        # Return -1 to signal zero-padding needed
        return -1
    elif boundary_mode == "reflect":
        # Mirror padding
        if z_index < z_min:
            return z_min + (z_min - z_index - 1)
        else:  # z_index >= z_max
            return z_max - 1 - (z_index - z_max)
    elif boundary_mode == "skip":
        # Return -1 to signal skip
        return -1
    else:
        raise ValueError(f"Unknown boundary_mode: {boundary_mode}")


def compute_channel_count(
    operations: List[ChannelOperation],
    input_channels: int
) -> int:
    """
    Compute total output channel count.
    
    Parameters
    ----------
    operations : List[ChannelOperation]
        List of channel operations
    input_channels : int
        Number of input channels
        
    Returns
    -------
    int
        Total output channels = input_channels × sum(operation.output_channels)
    """
    total_z_channels = sum(op.output_channels for op in operations)
    return input_channels * total_z_channels


@dataclass
class PatchExtraction:
    """Single patch extraction specification."""
    patch_idx: int  # Linear patch index
    n: int  # Image index
    z0: int  # Center z-slice
    y_start: Optional[int] = None  # Y start (if 2D quilting)
    y_stop: Optional[int] = None  # Y stop
    x_start: Optional[int] = None  # X start (if 2D quilting)
    x_stop: Optional[int] = None  # X stop
    color_y_idx: int = 0  # Color group for Y
    color_x_idx: int = 0  # Color group for X
    required_z_indices: List[int] = field(default_factory=list)  # All z-slices needed


@dataclass
class ExtractionPlan:
    """Plan for extracting patches."""
    patches: List[PatchExtraction]
    color_groups: Dict[Tuple[int, int], List[int]]  # (color_y, color_x) -> patch indices
    total_patches: int
    
    def get_patches_for_color(self, color_y: int, color_x: int) -> List[PatchExtraction]:
        """
        Get all patches for a specific color group.
        
        Parameters
        ----------
        color_y : int
            Y color group index
        color_x : int
            X color group index
            
        Returns
        -------
        List[PatchExtraction]
            All patches in this color group
        """
        patch_indices = self.color_groups.get((color_y, color_x), [])
        return [self.patches[idx] for idx in patch_indices]
    
    def serialize(self) -> Dict:
        """
        Serialize plan for distributed processing.
        
        Returns
        -------
        Dict
            Serializable representation of the plan
        """
        return {
            'patches': [
                {
                    'patch_idx': p.patch_idx,
                    'n': p.n,
                    'z0': p.z0,
                    'y_start': p.y_start,
                    'y_stop': p.y_stop,
                    'x_start': p.x_start,
                    'x_stop': p.x_stop,
                    'color_y_idx': p.color_y_idx,
                    'color_x_idx': p.color_x_idx,
                    'required_z_indices': p.required_z_indices,
                }
                for p in self.patches
            ],
            'color_groups': {f"{k[0]},{k[1]}": v for k, v in self.color_groups.items()},
            'total_patches': self.total_patches,
        }
    
    @classmethod
    def deserialize(cls, data: Dict) -> 'ExtractionPlan':
        """
        Deserialize plan from dict.
        
        Parameters
        ----------
        data : Dict
            Serialized plan data
            
        Returns
        -------
        ExtractionPlan
            Reconstructed plan
        """
        patches = [
            PatchExtraction(
                patch_idx=p['patch_idx'],
                n=p['n'],
                z0=p['z0'],
                y_start=p.get('y_start'),
                y_stop=p.get('y_stop'),
                x_start=p.get('x_start'),
                x_stop=p.get('x_stop'),
                color_y_idx=p['color_y_idx'],
                color_x_idx=p['color_x_idx'],
                required_z_indices=p['required_z_indices'],
            )
            for p in data['patches']
        ]
        
        color_groups = {
            tuple(map(int, k.split(','))): v
            for k, v in data['color_groups'].items()
        }
        
        return cls(
            patches=patches,
            color_groups=color_groups,
            total_patches=data['total_patches']
        )


@dataclass
class StitchingPlan:
    """Plan for stitching patches back together."""
    output_shape: Tuple[int, ...]  # (N, C', Y, X) or (N, C', Z, Y, X)
    patch_mappings: Dict[int, Dict]  # patch_idx -> {output_location, weight, ...}
    color_groups: Dict[Tuple[int, int], List[int]]  # Same as extraction plan
    
    def get_stitch_order(self, color_y: int, color_x: int) -> List[int]:
        """
        Get patch indices to stitch for a color group.
        
        Parameters
        ----------
        color_y : int
            Y color group index
        color_x : int
            X color group index
            
        Returns
        -------
        List[int]
            Patch indices to stitch for this color group
        """
        return self.color_groups.get((color_y, color_x), [])
    
    def serialize(self) -> Dict:
        """
        Serialize plan for distributed processing.
        
        Returns
        -------
        Dict
            Serializable representation of the plan
        """
        return {
            'output_shape': list(self.output_shape),
            'patch_mappings': self.patch_mappings,
            'color_groups': {f"{k[0]},{k[1]}": v for k, v in self.color_groups.items()},
        }
    
    @classmethod
    def deserialize(cls, data: Dict) -> 'StitchingPlan':
        """
        Deserialize plan from dict.
        
        Parameters
        ----------
        data : Dict
            Serialized plan data
            
        Returns
        -------
        StitchingPlan
            Reconstructed plan
        """
        color_groups = {
            tuple(map(int, k.split(','))): v
            for k, v in data['color_groups'].items()
        }
        
        return cls(
            output_shape=tuple(data['output_shape']),
            patch_mappings=data['patch_mappings'],
            color_groups=color_groups,
        )


class NCZYX25DQuilt:
    """
    Converts 3D data (N, C, Z, Y, X) to 2.5D multi-channel data.
    
    Supports selective slicing, two accumulation modes (2D planes or 3D stack),
    and flexible channel specifications via ZOperation enum.
    
    Works with tensor-like objects that support indexing and return torch.Tensor.
    
    Integrates with 2D patch extraction:
    - extract_patch_pairs(): Extract patch pairs from converted 2.5D data
    - extract_overlapping_pixels(): Extract overlapping pixels from patch pairs
    """
    
    def __init__(
        self,
        data_source: Union[torch.Tensor, "TensorLike3D"],
        channel_spec: Union[
            Dict[str, Union[List[int], List[List[int]]]],
            Dict[ZOperation, Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]],
        ],
        boundary_mode: str = "clamp",
        accumulation_mode: str = "2d",
        z_slices: Optional[Union[slice, List[int]]] = None,
        group_by_operation: bool = False,
    ):
        """
        Parameters
        ----------
        data_source : Union[torch.Tensor, TensorLike3D]
            Input 3D data of shape (N, C, Z, Y, X).
            Can be torch.Tensor or TensorLike3D wrapper around backend.
        channel_spec : Dict
            Channel specification with ZOperation enum or string keys
        boundary_mode : str
            Z-boundary handling: "clamp", "zero", "reflect", "skip"
        accumulation_mode : str
            "2d": Flatten to (N, C', Y, X) - 2D planes
            "3d": Keep as (N, C', Z, Y, X) - 3D stack
        z_slices : Optional[Union[slice, List[int]]]
            Which z-slices to process. None = all slices.
        group_by_operation : bool
            If True, group output channels by operation type
        """
        # Handle data source - convert TensorLike3D or use torch.Tensor directly
        try:
            from qlty.backends_2_5D import TensorLike3D as _TensorLike3D
            is_tensor_like = isinstance(data_source, _TensorLike3D)
        except (ImportError, TypeError):
            is_tensor_like = False
        
        if is_tensor_like:
            self.data_source = data_source
            shape = data_source.shape
        elif isinstance(data_source, torch.Tensor):
            self.data_source = data_source
            shape = data_source.shape
        else:
            raise TypeError(
                f"data_source must be torch.Tensor or TensorLike3D, got {type(data_source)}"
            )
        
        # Validate inputs
        if len(shape) != 5:
            raise ValueError(f"data_source must be 5D (N, C, Z, Y, X), got shape {shape}")
        
        if accumulation_mode not in ("2d", "3d"):
            raise ValueError(f"accumulation_mode must be '2d' or '3d', got '{accumulation_mode}'")
        
        if boundary_mode not in ("clamp", "zero", "reflect", "skip"):
            raise ValueError(
                f"boundary_mode must be 'clamp', 'zero', 'reflect', or 'skip', got '{boundary_mode}'"
            )
        
        # Store inputs
        self.boundary_mode = boundary_mode
        self.accumulation_mode = accumulation_mode
        self.group_by_operation = group_by_operation
        
        # Parse channel specification
        self.operations = parse_channel_spec(channel_spec)
        
        # Get data shape
        N, C, Z, Y, X = shape
        self.N = N
        self.C = C
        self.Z = Z
        self.Y = Y
        self.X = X
        
        # Process z_slices parameter
        if z_slices is None:
            self.z_indices = list(range(Z))
        elif isinstance(z_slices, slice):
            self.z_indices = list(range(Z)[z_slices])
        elif isinstance(z_slices, (list, tuple)):
            self.z_indices = [int(z) for z in z_slices]
            # Validate indices
            for z in self.z_indices:
                if z < 0 or z >= Z:
                    raise ValueError(f"z_slices contains invalid index {z} (Z={Z})")
        else:
            raise ValueError(f"z_slices must be slice, list, or None, got {type(z_slices)}")
        
        if not self.z_indices:
            raise ValueError("z_slices results in empty list")
        
        # Compute output channel count
        self.output_channels = compute_channel_count(self.operations, C)
        
        # Compute output shape
        if accumulation_mode == "2d":
            self.output_shape = (N, self.output_channels, Y, X)
        else:  # "3d"
            self.output_shape = (N, self.output_channels, len(self.z_indices), Y, X)
    
    def _apply_operation(
        self,
        data: torch.Tensor,
        operation: ChannelOperation,
        z0: int
    ) -> torch.Tensor:
        """
        Apply a single channel operation at z-slice z0.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (C, Z, Y, X) or (C, Y, X) for single z
        operation : ChannelOperation
            Operation to apply
        z0 : int
            Center z-slice index
            
        Returns
        -------
        torch.Tensor
            Output channels of shape (output_channels, Y, X)
        """
        C, Z, Y, X = data.shape
        results = []
        
        if operation.op_type == ZOperation.DIRECT:
            # DIRECT: Extract specific z-slices
            offsets = operation.offsets
            for offset in offsets:
                z_idx = z0 + offset
                z_clamped = apply_boundary_mode(z_idx, 0, Z, self.boundary_mode)
                
                if z_clamped == -1:
                    # Zero padding or skip
                    if self.boundary_mode == "zero":
                        channel = torch.zeros((C, Y, X), dtype=data.dtype, device=data.device)
                    else:  # skip
                        continue  # Skip this channel
                else:
                    channel = data[:, z_clamped, :, :]
                
                results.append(channel)
                
        elif operation.op_type == ZOperation.MEAN:
            # MEAN: Average multiple z-slices
            offset_groups = operation.offsets
            for offset_group in offset_groups:
                slices = []
                for offset in offset_group:
                    z_idx = z0 + offset
                    z_clamped = apply_boundary_mode(z_idx, 0, Z, self.boundary_mode)
                    
                    if z_clamped == -1:
                        if self.boundary_mode == "zero":
                            slices.append(torch.zeros((C, Y, X), dtype=data.dtype, device=data.device))
                        # skip mode: just don't include this slice
                    else:
                        slices.append(data[:, z_clamped, :, :])
                
                if slices:
                    # Compute mean
                    stacked = torch.stack(slices, dim=0)  # (num_slices, C, Y, X)
                    mean_channel = torch.mean(stacked, dim=0)  # (C, Y, X)
                    results.append(mean_channel)
        
        if not results:
            raise ValueError(f"Operation {operation} produced no valid channels (all skipped?)")
        
        # Stack results: (num_channels, C, Y, X)
        stacked = torch.stack(results, dim=0)
        # Reshape to (C * num_channels, Y, X) - interleave channels
        C_out = stacked.shape[0] * C
        return stacked.permute(1, 0, 2, 3).reshape(C_out, Y, X)
    
    def convert(self) -> torch.Tensor:
        """
        Convert 3D data to 2.5D according to channel specification.
        
        Returns
        -------
        torch.Tensor
            Converted data:
            - If accumulation_mode="2d": shape (N, C', Y, X)
            - If accumulation_mode="3d": shape (N, C', Z_selected, Y, X)
        """
        device = self.data_source.device
        dtype = self.data_source.dtype
        
        if self.accumulation_mode == "2d":
            # Flatten Z dimension into channels
            output = torch.zeros(
                (self.N, self.output_channels, self.Y, self.X),
                dtype=dtype,
                device=device
            )
            
            # For each z-slice, apply operations and accumulate
            for z0 in self.z_indices:
                # Load data for this z-slice and neighbors
                # Get all required z-indices for all operations
                all_z_indices = set()
                for op in self.operations:
                    z_min, z_max = op.get_required_z_range(z0)
                    all_z_indices.update(range(z_min, z_max))
                
                # Clamp z indices to valid range
                all_z_indices = [z for z in all_z_indices if 0 <= z < self.Z]
                if not all_z_indices:
                    continue
                
                z_min_load = min(all_z_indices)
                z_max_load = max(all_z_indices) + 1
                data_chunk = self.data_source[:, :, z_min_load:z_max_load, :, :]  # (N, C, Z_chunk, Y, X)
                
                # Process each image
                for n in range(self.N):
                    # Apply all operations
                    channel_results = []
                    for op in self.operations:
                        op_result = self._apply_operation(
                            data_chunk[n],  # (C, Z_chunk, Y, X)
                            op,
                            z0 - z_min_load  # Adjust z0 relative to chunk
                        )
                        channel_results.append(op_result)
                    
                    # Concatenate all operation results
                    if channel_results:
                        combined = torch.cat(channel_results, dim=0)  # (C', Y, X)
                        # Accumulate (will average at the end)
                        output[n] += combined
            
            # Average across all processed z-slices
            if len(self.z_indices) > 0:
                output = output / len(self.z_indices)
            
            return output
            
        else:  # "3d"
            # Keep Z dimension separate
            Z_selected = len(self.z_indices)
            output = torch.zeros(
                (self.N, self.output_channels, Z_selected, self.Y, self.X),
                dtype=dtype,
                device=device
            )
            
            for z_idx, z0 in enumerate(self.z_indices):
                # Get all required z-indices for all operations
                all_z_indices = set()
                for op in self.operations:
                    z_min, z_max = op.get_required_z_range(z0)
                    all_z_indices.update(range(z_min, z_max))
                
                # Clamp z indices to valid range
                all_z_indices = [z for z in all_z_indices if 0 <= z < self.Z]
                if not all_z_indices:
                    continue
                
                z_min_load = min(all_z_indices)
                z_max_load = max(all_z_indices) + 1
                data_chunk = self.data_source[:, :, z_min_load:z_max_load, :, :]  # (N, C, Z_chunk, Y, X)
                
                # Process each image
                for n in range(self.N):
                    # Apply all operations
                    channel_results = []
                    for op in self.operations:
                        op_result = self._apply_operation(
                            data_chunk[n],  # (C, Z_chunk, Y, X)
                            op,
                            z0 - z_min_load  # Adjust z0 relative to chunk
                        )
                        channel_results.append(op_result)
                    
                    # Concatenate all operation results
                    if channel_results:
                        combined = torch.cat(channel_results, dim=0)  # (C', Y, X)
                        output[n, :, z_idx, :, :] = combined
            
            return output
    
    def get_channel_metadata(self) -> List[Dict]:
        """
        Returns metadata for each output channel.
        
        Returns
        -------
        List[Dict]
            List of metadata dicts, one per output channel
        """
        metadata = []
        channel_idx = 0
        
        for input_c in range(self.C):
            for op in self.operations:
                op_name = op.name or op.op_type.name.lower()
                
                if op.op_type == ZOperation.DIRECT:
                    for offset in op.offsets:
                        metadata.append({
                            'channel_index': channel_idx,
                            'input_channel': input_c,
                            'operation': op_name,
                            'operation_type': op.op_type.name,
                            'offsets': (offset,),
                            'z0': None,  # Will be set per z-slice
                        })
                        channel_idx += 1
                        
                elif op.op_type == ZOperation.MEAN:
                    for offset_group in op.offsets:
                        metadata.append({
                            'channel_index': channel_idx,
                            'input_channel': input_c,
                            'operation': op_name,
                            'operation_type': op.op_type.name,
                            'offsets': offset_group,
                            'z0': None,  # Will be set per z-slice
                        })
                        channel_idx += 1
        
        return metadata
    
    def validate_spec(self) -> Tuple[bool, List[str]]:
        """
        Validate channel specification against data source.
        
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_warnings_or_errors)
        """
        errors = []
        warnings = []
        
        # Check that operations are valid
        for op in self.operations:
            if op.op_type == ZOperation.DIRECT:
                offsets = op.offsets
                for offset in offsets:
                    # Check if offsets are reasonable
                    if abs(offset) > self.Z:
                        warnings.append(
                            f"DIRECT offset {offset} is larger than Z dimension ({self.Z})"
                        )
            elif op.op_type == ZOperation.MEAN:
                for offset_group in op.offsets:
                    for offset in offset_group:
                        if abs(offset) > self.Z:
                            warnings.append(
                                f"MEAN offset {offset} is larger than Z dimension ({self.Z})"
                            )
        
        # Check z_slices
        if self.z_indices:
            if max(self.z_indices) >= self.Z:
                errors.append(f"z_slices contains index >= Z ({self.Z})")
            if min(self.z_indices) < 0:
                errors.append("z_slices contains negative index")
        
        is_valid = len(errors) == 0
        return is_valid, errors + warnings
    
    def to_ncyx_quilt(self, **quilt_kwargs):
        """
        Convert to 2D and create NCYXQuilt in one step.
        Returns configured NCYXQuilt with converted data.
        
        Parameters
        ----------
        **quilt_kwargs
            Arguments passed to NCYXQuilt constructor (window, step, border, etc.)
        
        Returns
        -------
        NCYXQuilt
            Configured 2D quilt with converted 2.5D data
        
        Notes
        -----
        This method converts the 3D data to 2.5D first, then creates a 2D quilt.
        The accumulation_mode must be "2d" for this to work.
        """
        if self.accumulation_mode != "2d":
            raise ValueError(
                "to_ncyx_quilt requires accumulation_mode='2d'. "
                f"Current mode: {self.accumulation_mode}"
            )
        
        from qlty.qlty2D import NCYXQuilt
        
        # Convert to 2.5D
        converted = self.convert()  # (N, C', Y, X)
        
        # Create 2D quilt
        N, C, Y, X = converted.shape
        return NCYXQuilt(
            Y=Y,
            X=X,
            **quilt_kwargs
        )
    
    def extract_patch_pairs(
        self,
        window: Tuple[int, int],
        num_patches: int,
        delta_range: Tuple[float, float],
        random_seed: Optional[int] = None,
        rotation_choices: Optional[List[int]] = None,
    ):
        """
        Extract patch pairs from converted 2.5D data using 2D interface.
        
        This method converts 3D data to 2.5D, then extracts patch pairs using
        the 2D extract_patch_pairs function.
        
        Parameters
        ----------
        window : Tuple[int, int]
            Window shape (U, V) for patches
        num_patches : int
            Number of patch pairs P to extract per image
        delta_range : Tuple[float, float]
            Range (low, high) for Euclidean distance of displacement vectors
        random_seed : Optional[int]
            Random seed for reproducibility
        rotation_choices : Optional[List[int]]
            Allowed quarter-turn rotations (0, 1, 2, 3)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (patches1, patches2, deltas, rotations) as returned by extract_patch_pairs
        
        Notes
        -----
        This requires accumulation_mode="2d" to work with the 2D interface.
        """
        if self.accumulation_mode != "2d":
            raise ValueError(
                "extract_patch_pairs requires accumulation_mode='2d'. "
                f"Current mode: {self.accumulation_mode}"
            )
        
        from qlty.patch_pairs_2d import extract_patch_pairs
        
        # Convert to 2.5D
        converted = self.convert()  # (N, C', Y, X)
        
        # Extract patch pairs using 2D interface
        return extract_patch_pairs(
            converted,
            window=window,
            num_patches=num_patches,
            delta_range=delta_range,
            random_seed=random_seed,
            rotation_choices=rotation_choices,
        )
    
    def extract_overlapping_pixels(
        self,
        window: Tuple[int, int],
        num_patches: int,
        delta_range: Tuple[float, float],
        random_seed: Optional[int] = None,
        rotation_choices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract overlapping pixels from patch pairs using 2D interface.
        
        This method combines extract_patch_pairs and extract_overlapping_pixels:
        1. Converts 3D data to 2.5D
        2. Extracts patch pairs
        3. Extracts overlapping pixels from the patch pairs
        
        Parameters
        ----------
        window : Tuple[int, int]
            Window shape (U, V) for patches
        num_patches : int
            Number of patch pairs P to extract per image
        delta_range : Tuple[float, float]
            Range (low, high) for Euclidean distance of displacement vectors
        random_seed : Optional[int]
            Random seed for reproducibility
        rotation_choices : Optional[List[int]]
            Allowed quarter-turn rotations (0, 1, 2, 3)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (overlapping1, overlapping2) where:
            - overlapping1: Overlapping pixel values from patches1, shape (K, C')
            - overlapping2: Overlapping pixel values from patches2, shape (K, C')
            where K is the total number of overlapping pixels across all patch pairs,
            and corresponding pixels are at the same index in both tensors.
        
        Notes
        -----
        This requires accumulation_mode="2d" to work with the 2D interface.
        
        Examples
        --------
        >>> quilt = NCZYX25DQuilt(data, {'direct': [0]}, accumulation_mode="2d")
        >>> overlapping1, overlapping2 = quilt.extract_overlapping_pixels(
        ...     window=(32, 32),
        ...     num_patches=100,
        ...     delta_range=(8.0, 16.0)
        ... )
        >>> # overlapping1[i] and overlapping2[i] correspond to the same spatial location
        """
        if self.accumulation_mode != "2d":
            raise ValueError(
                "extract_overlapping_pixels requires accumulation_mode='2d'. "
                f"Current mode: {self.accumulation_mode}"
            )
        
        from qlty.patch_pairs_2d import extract_overlapping_pixels
        
        # Extract patch pairs first
        patches1, patches2, deltas, rotations = self.extract_patch_pairs(
            window=window,
            num_patches=num_patches,
            delta_range=delta_range,
            random_seed=random_seed,
            rotation_choices=rotation_choices,
        )
        
        # Extract overlapping pixels
        return extract_overlapping_pixels(
            patches1=patches1,
            patches2=patches2,
            deltas=deltas,
            rotations=rotations,
        )
    
    def create_extraction_plan(
        self,
        color_y_mod: Optional[int] = None,
        color_x_mod: Optional[int] = None,
        window: Optional[Tuple[int, int]] = None,
        step: Optional[Tuple[int, int]] = None,
    ) -> ExtractionPlan:
        """
        Create extraction plan specifying what to extract and when.
        Includes color group assignments for parallelization.
        
        Parameters
        ----------
        color_y_mod : Optional[int]
            Color modulo for Y dimension (for parallelization).
            If None and window/step provided, computed automatically.
        color_x_mod : Optional[int]
            Color modulo for X dimension (for parallelization).
            If None and window/step provided, computed automatically.
        window : Optional[Tuple[int, int]]
            Window size (Y, X) for 2D quilting. If provided, creates patches.
        step : Optional[Tuple[int, int]]
            Step size (Y, X) for 2D quilting. Required if window provided.
        
        Returns
        -------
        ExtractionPlan
            Plan with patch indices, z-slices, and color groups
        """
        import math
        
        patches = []
        color_groups: Dict[Tuple[int, int], List[int]] = {}
        patch_idx = 0
        
        # Determine if we're doing 2D quilting
        doing_2d_quilting = window is not None and step is not None
        
        if doing_2d_quilting:
            # Compute number of patches in Y and X (similar to compute_chunk_times)
            def compute_steps(dim_size, win_size, step_size):
                full_steps = (dim_size - win_size) // step_size
                if dim_size > full_steps * step_size + win_size:
                    return full_steps + 2
                else:
                    return full_steps + 1
            
            nY = compute_steps(self.Y, window[0], step[0])
            nX = compute_steps(self.X, window[1], step[1])
            
            # Compute color mods if not provided
            if color_y_mod is None:
                color_y_mod = max(1, math.ceil(window[0] / step[0]))
                color_y_mod = min(color_y_mod, nY) if nY > 0 else 1
            if color_x_mod is None:
                color_x_mod = max(1, math.ceil(window[1] / step[1]))
                color_x_mod = min(color_x_mod, nX) if nX > 0 else 1
        else:
            # No 2D quilting - just process full Y, X
            nY = 1
            nX = 1
            color_y_mod = color_y_mod or 1
            color_x_mod = color_x_mod or 1
        
        # For each image, z-slice, and patch position
        for n in range(self.N):
            for z0 in self.z_indices:
                # Get all required z-indices for all operations
                all_z_indices = set()
                for op in self.operations:
                    z_min, z_max = op.get_required_z_range(z0)
                    all_z_indices.update(range(z_min, z_max))
                
                # Clamp to valid range
                all_z_indices = sorted([z for z in all_z_indices if 0 <= z < self.Z])
                
                if not all_z_indices:
                    continue
                
                for yy in range(nY):
                    for xx in range(nX):
                        # Compute patch boundaries
                        if doing_2d_quilting:
                            y_start = min(yy * step[0], self.Y - window[0])
                            x_start = min(xx * step[1], self.X - window[1])
                            y_stop = y_start + window[0]
                            x_stop = x_start + window[1]
                        else:
                            y_start = None
                            x_start = None
                            y_stop = None
                            x_stop = None
                        
                        # Compute color group
                        color_y_idx = yy % color_y_mod
                        color_x_idx = xx % color_x_mod
                        
                        # Create patch extraction
                        patch = PatchExtraction(
                            patch_idx=patch_idx,
                            n=n,
                            z0=z0,
                            y_start=y_start,
                            y_stop=y_stop,
                            x_start=x_start,
                            x_stop=x_stop,
                            color_y_idx=color_y_idx,
                            color_x_idx=color_x_idx,
                            required_z_indices=all_z_indices,
                        )
                        
                        patches.append(patch)
                        
                        # Add to color group
                        color_key = (color_y_idx, color_x_idx)
                        if color_key not in color_groups:
                            color_groups[color_key] = []
                        color_groups[color_key].append(patch_idx)
                        
                        patch_idx += 1
        
        return ExtractionPlan(
            patches=patches,
            color_groups=color_groups,
            total_patches=len(patches)
        )
    
    def create_stitching_plan(
        self,
        extraction_plan: Optional[ExtractionPlan] = None,
        window: Optional[Tuple[int, int]] = None,
        step: Optional[Tuple[int, int]] = None,
    ) -> StitchingPlan:
        """
        Create stitching plan specifying how to stitch back together.
        Coordinates accumulation order and color groups.
        
        Parameters
        ----------
        extraction_plan : Optional[ExtractionPlan]
            Extraction plan to base stitching on. If None, creates a new one.
        window : Optional[Tuple[int, int]]
            Window size for 2D quilting (if applicable)
        step : Optional[Tuple[int, int]]
            Step size for 2D quilting (if applicable)
        
        Returns
        -------
        StitchingPlan
            Plan with output locations and accumulation order
        """
        if extraction_plan is None:
            extraction_plan = self.create_extraction_plan(window=window, step=step)
        
        patch_mappings = {}
        
        # Create mappings for each patch
        for patch in extraction_plan.patches:
            if patch.y_start is not None:
                # 2D quilting patch
                mapping = {
                    'n': patch.n,
                    'y_start': patch.y_start,
                    'y_stop': patch.y_stop,
                    'x_start': patch.x_start,
                    'x_stop': patch.x_stop,
                    'z_idx': None if self.accumulation_mode == "2d" else self.z_indices.index(patch.z0),
                }
            else:
                # Full image patch
                mapping = {
                    'n': patch.n,
                    'y_start': None,
                    'y_stop': None,
                    'x_start': None,
                    'x_stop': None,
                    'z_idx': None if self.accumulation_mode == "2d" else self.z_indices.index(patch.z0),
                }
            
            patch_mappings[patch.patch_idx] = mapping
        
        return StitchingPlan(
            output_shape=self.output_shape,
            patch_mappings=patch_mappings,
            color_groups=extraction_plan.color_groups,
        )

