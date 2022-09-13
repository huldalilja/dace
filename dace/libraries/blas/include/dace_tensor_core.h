// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <mma.h>
#include <string>     // std::to_string

namespace dace {

namespace blas {

static void CheckCudaError(cudaError_t const& status) {
  if (status != cudaSuccess) {
    throw std::runtime_error("cuda failed with error code: " +
                             std::to_string(status));
  }
}

/**
 * Class for WMMA constants.
 **/
class _WMMAConstants {
 public:
  int const WMMA_M = 16;

  _WMMAConstants(int device) {
    if (cudaSetDevice(device) != cudaSuccess) {
      throw std::runtime_error("Failed to set CUDA device.");
    }
  }

  _WMMAConstants(_WMMAConstants const&) = delete;

  ~_WMMAsConstants() {
  }

  _WMMAConstants& operator=(_WMMAConstants const&) = delete;

  void CheckError(cudaError_t const& status) {
    if (status != cudaSuccess) {
      throw std::runtime_error("cuda failed with error code: " +
                               std::to_string(status));
    }
  }
};

/**
 * CUBLAS wrapper class for DaCe. Once constructed, the class can be used to
 * get or create a CUBLAS library handle (cublasHandle_t) for a given GPU ID,
 * or get pre-allocated constants (see ``_CublasConstants`` class) for CUBLAS
 * calls.
 * The class is constructed when the CUBLAS DaCe library is used.
 **/
class CublasHandle {
 public:
  CublasHandle() = default;
  CublasHandle(CublasHandle const&) = delete;

  cublasHandle_t& Get(int device) {
    auto f = handles_.find(device);
    if (f == handles_.end()) {
      // Lazily construct new cuBLAS handle if the specified key does not yet
      // exist
      auto handle = CreateCublasHandle(device);
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
      f = handles_.emplace(device, handle).first;
    }
    return f->second;
  }

  _CublasConstants& Constants(int device) {
    auto f = constants_.find(device);
    if (f == constants_.end()) {
      // Lazily construct new cuBLAS constants
      f = constants_.emplace(device, device).first;
    }
    return f->second;
  }

  ~CublasHandle() {
    for (auto& h : handles_) {
      CheckCublasError(cublasDestroy(h.second));
    }
  }

  CublasHandle& operator=(CublasHandle const&) = delete;

  std::unordered_map<int, cublasHandle_t> handles_;
  std::unordered_map<int, _CublasConstants> constants_;
};

}  // namespace blas

}  // namespace dace
