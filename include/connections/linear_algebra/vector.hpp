#pragma once

#include <cstddef>

#include "connections/network/arena.hpp"
#include "connections/util/concepts/types.hpp"

namespace cntns {
template <size_t dim_s, ArenaType arena_e>
class Vec;

template <size_t rows, size_t cols, ArenaType arena_e>
class Matrix;
}  // namespace cntns

#include "cpu/vector.hpp"

#ifdef CNTNS_USE_CUDA

#include "gpu/vector.cuh"

#else

namespace cntns {
template <size_t dim_s>
class Vec<dim_s, ArenaType::GPU> {
};
}  // namespace cntns

#endif