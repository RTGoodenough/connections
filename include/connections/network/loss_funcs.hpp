#pragma once

#include <cmath>
#include <numeric>

#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"

#ifdef CNTNS_USE_CUDA
#include "connections/linear_algebra/gpu/cuda/loss.cuh"
#endif

namespace cntns {

struct MSE {
  template <size_t dim_s>
  [[nodiscard]] static auto error(Vec<dim_s, ArenaType::CPU> const& correct,
                                  Vec<dim_s, ArenaType::CPU> const& result)
      -> Vec<dim_s, ArenaType::CPU>
  {
    return correct - result;
  }

  template <size_t dim_s>
  [[nodiscard]] static auto loss(Vec<dim_s, ArenaType::CPU> const& correct,
                                 Vec<dim_s, ArenaType::CPU> const& result)
      -> float
  {
    constexpr float            HALF = 0.5;
    Vec<dim_s, ArenaType::CPU> error = correct - result;
    return (HALF * dim_s) *
           std::inner_product(error.begin(), error.end(), error.begin(), 0.0F);
  }

#ifdef CNTNS_USE_CUDA
  template <size_t dim_s>
  [[nodiscard]] static auto error(Vec<dim_s, ArenaType::GPU> const& correct,
                                  Vec<dim_s, ArenaType::GPU> const& result)
      -> Vec<dim_s, ArenaType::GPU>
  {
    return correct - result;
  }

  template <size_t dim_s>
  [[nodiscard]] static auto loss(Vec<dim_s, ArenaType::GPU> const& correct,
                                 Vec<dim_s, ArenaType::GPU> const& result)
      -> float
  {
    return util::get_return<float>([&](float* answer) {
      quadratic_cost_kernel<<<
          std::ceil(Vec<dim_s, ArenaType::GPU>::SIZE / 512.0), 512>>>(
          correct.data(), result.data(), answer, result.SIZE);
    });
  }
#endif
};

struct CrossEntropy {
  template <size_t dim_s>
  [[nodiscard]] static auto error(Vec<dim_s, ArenaType::CPU> const& correct,
                                  Vec<dim_s, ArenaType::CPU> const& result)
      -> Vec<dim_s, ArenaType::CPU>
  {
    return correct - result;
  }

  template <size_t dim_s>
  [[nodiscard]] static auto loss(Vec<dim_s, ArenaType::CPU> const& correct,
                                 Vec<dim_s, ArenaType::CPU> const& result)
      -> float
  {
    float lossVal{};

    for ( size_t i = 0; i < dim_s; ++i )
      lossVal += (correct[i] * std::log(result[i]));

    return -lossVal;
  }

#ifdef CNTNS_USE_CUDA
  template <size_t dim_s>
  [[nodiscard]] static auto error(Vec<dim_s, ArenaType::GPU> const& correct,
                                  Vec<dim_s, ArenaType::GPU> const& result)
      -> Vec<dim_s, ArenaType::GPU>
  {
    return correct - result;
  }

  template <size_t dim_s>
  [[nodiscard]] static auto loss(Vec<dim_s, ArenaType::GPU> const& correct,
                                 Vec<dim_s, ArenaType::GPU> const& result)
      -> float
  {
    return util::get_return<float>([&](float* answer) {
      cross_entropy<<<std::ceil(Vec<dim_s, ArenaType::GPU>::SIZE / 512.0),
                      512>>>(correct.data(), result.data(), answer,
                             result.SIZE);
    });
  }
#endif
};

}  // namespace cntns