#pragma once

#include <cmath>
#include <numeric>
#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"
namespace cntns {

struct MSE {
  template <util::Numeric data_t, std::size_t dim_s>
  [[nodiscard]] static auto error(
      Vec<data_t, dim_s, ArenaType::CPU> const& correct,
      Vec<data_t, dim_s, ArenaType::CPU> const& result)
      -> Vec<data_t, dim_s, ArenaType::CPU>
  {
    return correct - result;
  }

  template <util::Numeric data_t, std::size_t dim_s>
  [[nodiscard]] static auto loss(
      Vec<data_t, dim_s, ArenaType::CPU> const& correct,
      Vec<data_t, dim_s, ArenaType::CPU> const& result) -> double
  {
    constexpr data_t                   HALF = 0.5;
    Vec<data_t, dim_s, ArenaType::CPU> error = correct - result;
    return (HALF * dim_s) *
           std::inner_product(error.begin(), error.end(), error.begin(), 0.0F);
  }
};

struct CrossEntropy {
  template <util::Numeric data_t, std::size_t dim_s>
  [[nodiscard]] static auto error(
      Vec<data_t, dim_s, ArenaType::CPU> const& correct,
      Vec<data_t, dim_s, ArenaType::CPU> const& result)
      -> Vec<data_t, dim_s, ArenaType::CPU>
  {
    return correct - result;
  }

  template <util::Numeric data_t, std::size_t dim_s>
  [[nodiscard]] static auto loss(
      Vec<data_t, dim_s, ArenaType::CPU> const& correct,
      Vec<data_t, dim_s, ArenaType::CPU> const& result) -> double
  {
    double lossVal{};

    for ( size_t i = 0; i < dim_s; ++i )
      lossVal += (correct[i] * std::log(result[i]));

    return -lossVal;
  }
};

}  // namespace cntns