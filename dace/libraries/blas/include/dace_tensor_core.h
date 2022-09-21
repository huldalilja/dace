// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>

namespace dace {

namespace blas {

static void CheckCudaError(cudaError_t const& status) {
  if (status != cudaSuccess) {
    throw std::runtime_error("cuda failed with error code: " +
                             std::to_string(status));
  }
}

// Maybe add constants like cuBLAS

}  // namespace blas

}  // namespace dace
