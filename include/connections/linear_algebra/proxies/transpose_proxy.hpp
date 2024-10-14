#pragma once

#ifdef CNTNS_USE_CUDA

#include "connections/linear_algebra/gpu/cuda/matrix.cuh"
#include "connections/linear_algebra/gpu/vector.cuh"

#include "connections/network/arena.hpp"

namespace cntns {

/**
 * @brief Proxy object for holding matrix transpose operations
 * 
 * @tparam rows 
 * @tparam cols 
 */
template <size_t rows, size_t cols>
struct TransposeProxy {
  float const* mat;

  [[nodiscard]] auto operator*(Vec<rows, ArenaType::GPU> const& input) const
      -> Vec<cols, ArenaType::GPU>
  {
    Vec<cols, ArenaType::GPU> result;
    mat_transpose_vec_mul_kernel<<<
        Matrix<rows, cols, ArenaType::GPU>::GRID_SIZE,
        Matrix<rows, cols, ArenaType::GPU>::BLOCK_SIZE>>>(
        mat, input.data(), result.data(), rows, cols);
    return result;
  }
};
}  // namespace cntns

#endif