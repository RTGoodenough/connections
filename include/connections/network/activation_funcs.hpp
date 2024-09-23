#pragma once

#include <algorithm>
#include <numeric>

#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"
#include "connections/util/concepts/types.hpp"

namespace cntns {

struct Sigmoid {
  template <util::Numeric data_t, size_t dim_s>
  [[nodiscard]] static auto eval(Vec<data_t, dim_s, ArenaType::CPU> const& vec)
  {
    Vec<data_t, dim_s, ArenaType::CPU> result;
    std::transform(vec.begin(), vec.end(), result.begin(),
                   [](auto value) { return 1.0F / (1.0F + std::exp(-value)); });
    return result;
  }

  template <util::Numeric data_t, size_t dim_s>
  [[nodiscard]] static auto derivative(
      Vec<data_t, dim_s, ArenaType::CPU> const& vec)
  {
    Vec<data_t, dim_s, ArenaType::CPU> result;
    std::transform(vec.begin(), vec.end(), result.begin(),
                   [](auto value) { return value * (1.0F - value); });
    return result;
  }
};

struct ReLu {
  template <util::Numeric data_t, size_t dim_s>
  [[nodiscard]] static auto eval(Vec<data_t, dim_s, ArenaType::CPU> const& vec)
  {
    Vec<data_t, dim_s, ArenaType::CPU> result;
    std::transform(vec.begin(), vec.end(), result.begin(),
                   [](auto value) { return value >= 0 ? value : 0.0; });
    return result;
  }

  template <util::Numeric data_t, size_t dim_s>
  [[nodiscard]] static auto derivative(
      Vec<data_t, dim_s, ArenaType::CPU> const& vec)
  {
    Vec<data_t, dim_s, ArenaType::CPU> result;
    std::transform(vec.begin(), vec.end(), result.begin(),
                   [](auto value) { return value >= 0 ? 1.0 : 0.0; });
    return result;
  }
};

struct Tanh {};

struct SoftMax {
  template <util::Numeric data_t, size_t dim_s>
  [[nodiscard]] static auto eval(Vec<data_t, dim_s, ArenaType::CPU> const& vec)
  {
    Vec<data_t, dim_s, ArenaType::CPU> prob;
    Vec<data_t, dim_s, ArenaType::CPU> exp;
    double max = *std::max_element(vec.begin(), vec.end());

    for ( size_t i = 0; i < dim_s; ++i ) {
      exp[i] = std::exp(vec[i] - max);
    }

    double expSum = std::accumulate(exp.begin(), exp.end(), 0.0);
    for ( size_t i = 0; i < dim_s; ++i ) {
      prob[i] = exp[i] / expSum;
    }

    return prob;
  }

  template <util::Numeric data_t, size_t dim_s>
  [[nodiscard]] static auto derivative(
      Vec<data_t, dim_s, ArenaType::CPU> const& vec)
  {
    return vec;
  }
};

}  // namespace cntns