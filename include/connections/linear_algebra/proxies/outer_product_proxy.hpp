#pragma once

#ifdef CNTNS_USE_CUDA

#include <cstddef>

namespace cntns {

/**
 * @brief Proxy object for holding GPU vector outer product operations
 * 
 * @tparam rows 
 * @tparam cols 
 */
template <size_t rows, size_t cols>
struct OuterProductProxy {
  float const* lhs;
  float const* rhs;
};
}  // namespace cntns

#endif