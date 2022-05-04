/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cpu/array_index_select_uvm.cuh
 * \brief Array index select GPU kernel implementation
 */

#ifndef DGL_ARRAY_CUDA_ARRAY_INDEX_SELECT_UVM_CUH_
#define DGL_ARRAY_CUDA_ARRAY_INDEX_SELECT_UVM_CUH_

#define CACHE_LINE_SIZE 128

namespace dgl {
namespace aten {
namespace impl {

/*  This is a cross-device access version of IndexSelectMultiKernel.
*   Since the memory access over PCIe is more sensitive to the
*   data access aligment (cacheline), we need a separate version here.
*/
template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernelAligned(
        const DType* const array,
        const int64_t num_feat,
        const IdType* const index,
        const int64_t length,
        const int64_t arr_len,
        DType* const out) {
  int64_t out_row = blockIdx.x*blockDim.y+threadIdx.y;

  const int64_t stride = blockDim.y*gridDim.x;

  while (out_row < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = index[out_row];
    assert(in_row >= 0 && in_row < arr_len);
    const int64_t idx_offset =
      ((uint64_t)(&array[in_row*num_feat]) % CACHE_LINE_SIZE) / sizeof(DType);
    col = col - idx_offset;
    while (col < num_feat) {
      if (col >= 0)
        out[out_row*num_feat+col] = array[in_row*num_feat+col];
      col += blockDim.x;
    }
    out_row += stride;
  }
}

template <typename DType, typename IdType>
__global__ void IndexSelectFromCacheMultiKernelAligned(
        const DType* const host_rows,
        const DType* const cached_rows,
        const int64_t num_feat,
        const IdType* const index,
        const IdType* const row_pos_map,
        const int64_t length,
        const int64_t host_len,
        const int64_t cached_len,
        DType* const out) {
  int64_t out_row = blockIdx.x*blockDim.y+threadIdx.y;

  const int64_t stride = blockDim.y*gridDim.x;

  while (out_row < length) {
    int64_t col = threadIdx.x;
    int64_t in_row = index[out_row];
    assert(in_row >= 0 && in_row < host_len + cached_len);
    int64_t pos = row_pos_map[in_row];
    const DType* array;
    if (pos >= 0) {
      array = cached_rows;
    } else {
      array = host_rows;
      pos = - pos - 2;
    }
    const int64_t idx_offset =
      ((uint64_t)(&array[pos*num_feat]) % CACHE_LINE_SIZE) / sizeof(DType);
    col = col - idx_offset;
    while (col < num_feat) {
      if (col >= 0)
        out[out_row*num_feat+col] = array[pos*num_feat+col];
      col += blockDim.x;
    }
    out_row += stride;
  }
}

template <typename DType, typename IdType>
__global__ void IndexSelectToCacheMultiKernelAligned(
        DType* host_rows,
        DType* cached_rows,
        const DType* const input,
        const int64_t num_feat,
        const IdType* const index,
        const IdType* const row_pos_map,
        const int64_t length,
        const int64_t host_len,
        const int64_t cached_len) {
  int64_t in_row = blockIdx.x*blockDim.y+threadIdx.y;

  const int64_t stride = blockDim.y*gridDim.x;

  while (in_row < length) {
    int64_t col = threadIdx.x;
    int64_t out_row = index[in_row];
    int64_t pos = row_pos_map[out_row];
    DType* array;
    if (pos >= 0) {
      array = cached_rows;
    } else {
      array = host_rows;
      pos = - pos - 2;
    }
    const int64_t idx_offset =
      ((uint64_t)(&array[pos*num_feat]) % CACHE_LINE_SIZE) / sizeof(DType);
    col = col - idx_offset;
    while (col < num_feat) {
      if (col >= 0)
        array[pos*num_feat+col] = input[in_row*num_feat+col];
      col += blockDim.x;
    }
    in_row += stride;
  }
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif
