#pragma once

#ifdef CNTNS_USE_CUDA

#include "connections/linear_algebra/gpu/matrix.hpp"
#include "connections/linear_algebra/gpu/vector.hpp"
#include "connections/network/arena.hpp"

namespace cntns {

/**
 * @brief Proxy object for holding matrix transpose operations
 * 
 * @tparam rows 
 * @tparam cols 
 */
template <typename data_t, size_t rows, size_t cols>
struct TransposeProxy {
  data_t const* mat;

  [[nodiscard]] auto operator*(Vec<data_t, rows, ArenaType::GPU> const& input)
      const -> Vec<data_t, cols, ArenaType::GPU>
  {
    Vec<data_t, cols, ArenaType::GPU> result;
    mat_transpose_vec_mul_kernel<<<Matrix<data_t, rows, cols>::GRID_SIZE,
                                   Matrix<data_t, rows, cols>::BLOCK_SIZE>>>(
        mat, input.data(), result.data(), rows, cols);
    return result;
  }
};
}  // namespace cntns

#endif