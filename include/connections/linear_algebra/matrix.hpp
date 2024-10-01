#pragma once

#include <cstddef>

#include "connections/network/arena.hpp"

namespace cntns {
template <size_t rows, size_t cols, ArenaType arena_e>
class Matrix;
}

#include "cpu/matrix.hpp"

#ifdef CNTNS_USE_CUDA

#include "gpu/matrix.cuh"

#else

namespace cntns {
template <size_t dim_s>
class Matrix<dim_s, ArenaType::GPU> {
};

#endif