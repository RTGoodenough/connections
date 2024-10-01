#pragma once

#include <cmath>
#include <numeric>
#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"

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
      -> double
  {
    constexpr double           HALF = 0.5;
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
      -> double
  {
    return 0;
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
      -> double
  {
    double lossVal{};

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
      -> double
  {
    return 0;
  }
#endif
};

}  // namespace cntns