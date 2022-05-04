"""Utility functions related to pinned memory tensors."""

from .. import backend as F
from .._ffi.function import _init_api

def pin_memory_inplace(tensor):
    """Register the tensor into pinned memory in-place (i.e. without copying)."""
    F.to_dgl_nd(tensor).pin_memory_()

def unpin_memory_inplace(tensor):
    """Unregister the tensor from pinned memory in-place (i.e. without copying)."""
    F.to_dgl_nd(tensor).unpin_memory_()

def gather_pinned_tensor_rows(tensor, rows):
    """Directly gather rows from a CPU tensor given an indices array on CUDA devices,
    and returns the result on the same CUDA device without copying.

    Parameters
    ----------
    tensor : Tensor
        The tensor.  Must be in pinned memory.
    rows : Tensor
        The rows to gather.  Must be a CUDA tensor.

    Returns
    -------
    Tensor
        The result with the same device as :attr:`rows`.
    """
    return F.from_dgl_nd(_CAPI_DGLIndexSelectCPUFromGPU(F.to_dgl_nd(tensor), F.to_dgl_nd(rows)))


def gather_cached_tensor_rows(host_rows, cached_rows, rows, row_pos_map):
    host_rows = F.to_dgl_nd(host_rows)
    cached_rows = F.to_dgl_nd(cached_rows)
    rows = F.to_dgl_nd(rows)
    row_pos_map = F.to_dgl_nd(row_pos_map)
    return F.from_dgl_nd(_CAPI_DGLIndexSelectFromCache(host_rows, cached_rows, rows, row_pos_map))

def scatter_cached_tensor_rows(host_rows, cached_rows, input, rows, row_pos_map):
    host_rows = F.to_dgl_nd(host_rows)
    cached_rows = F.to_dgl_nd(cached_rows)
    input = F.to_dgl_nd(input)
    rows = F.to_dgl_nd(rows)
    row_pos_map = F.to_dgl_nd(row_pos_map)
    _CAPI_DGLIndexSelectToCache(host_rows, cached_rows, input, rows, row_pos_map)


_init_api("dgl.ndarray.uvm", __name__)
