/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/uvm_array.cc
 * \brief DGL array utilities implementation
 */
#include <dgl/array.h>
#include <sstream>
#include "../c_api_common.h"
#include "./uvm_array_op.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {

NDArray IndexSelectCPUFromGPU(NDArray array, IdArray index) {
#ifdef DGL_USE_CUDA
  CHECK(array.IsPinned())
    << "Only the CPUPinned device type input array is supported";
  CHECK_EQ(index->ctx.device_type, kDLGPU)
    << "Only the GPU device type input index is supported";

  CHECK_GE(array->ndim, 1) << "Only support array with at least 1 dimension";
  CHECK_EQ(index->ndim, 1) << "Index array must be an 1D array.";
  ATEN_DTYPE_BITS_ONLY_SWITCH(array->dtype, DType, "values", {
    ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
      return impl::IndexSelectCPUFromGPU<DType, IdType>(array, index);
    });
  });
#endif
  LOG(FATAL) << "IndexSelectCPUFromGPU requires CUDA";
  // Should be unreachable
  return NDArray{};
}

NDArray IndexSelectFromCache(NDArray host_rows, NDArray cached_rows, IdArray index, IdArray row_pos_map) {
#ifdef DGL_USE_CUDA
  CHECK(host_rows->shape[0] == 0 || host_rows.IsPinned())
    << "Only the CPUPinned device type input array is supported";
  CHECK_EQ(cached_rows->ctx.device_type, kDLGPU)
    << "Only the GPU device type cached rows is supported";
  CHECK_EQ(index->ctx.device_type, kDLGPU)
    << "Only the GPU device type input index is supported";

  CHECK_EQ(index->ndim, 1) << "Index array must be an 1D array.";
  ATEN_DTYPE_BITS_ONLY_SWITCH(host_rows->dtype, DType, "values", {
    ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
      return impl::IndexSelectFromCache<DType, IdType>(host_rows, cached_rows, index, row_pos_map);
    });
  });
#endif
  LOG(FATAL) << "IndexSelectCPUFromCache requires CUDA";
  // Should be unreachable
  return NDArray{};
}

void IndexSelectToCache(NDArray host_rows, NDArray cached_rows, NDArray input, IdArray index, IdArray row_pos_map) {
#ifdef DGL_USE_CUDA
  CHECK(host_rows->shape[0] == 0 || host_rows.IsPinned())
    << "Only the CPUPinned device type input array is supported";
  CHECK_EQ(cached_rows->ctx.device_type, kDLGPU)
    << "Only the GPU device type cached rows is supported";
  CHECK_EQ(index->ctx.device_type, kDLGPU)
    << "Only the GPU device type input index is supported";

  CHECK_GE(input->ndim, 1) << "Only support array with at least 1 dimension";
  CHECK_EQ(index->ndim, 1) << "Index array must be an 1D array.";
  ATEN_DTYPE_BITS_ONLY_SWITCH(host_rows->dtype, DType, "values", {
    ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
      return impl::IndexSelectToCache<DType, IdType>(host_rows, cached_rows, input, index, row_pos_map);
    });
  });
#endif
  LOG(FATAL) << "IndexSelectCPUtoCache requires CUDA";
}

DGL_REGISTER_GLOBAL("ndarray.uvm._CAPI_DGLIndexSelectCPUFromGPU")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray array = args[0];
    IdArray index = args[1];
    *rv = IndexSelectCPUFromGPU(array, index);
  });

DGL_REGISTER_GLOBAL("ndarray.uvm._CAPI_DGLIndexSelectFromCache")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray host_rows = args[0];
    NDArray cached_rows = args[1];
    IdArray index = args[2];
    IdArray row_pos_map = args[3];
    *rv = IndexSelectFromCache(host_rows, cached_rows, index, row_pos_map);
  });

DGL_REGISTER_GLOBAL("ndarray.uvm._CAPI_DGLIndexSelectToCache")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray host_rows = args[0];
    NDArray cached_rows = args[1];
    NDArray input = args[2];
    IdArray index = args[3];
    IdArray row_pos_map = args[4];
    IndexSelectToCache(host_rows, cached_rows, input, index, row_pos_map);
  });

}  // namespace aten
}  // namespace dgl
