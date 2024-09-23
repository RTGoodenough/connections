#pragma once

#include <algorithm>

#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"
#include "connections/util/concepts/types.hpp"

namespace cntns {

struct Sigmoid {
  [[nodiscard]] static auto eval(double value) noexcept -> double { return 1 / (1 + std::exp(value)); }
  [[nodiscard]] static auto derivative(double value) noexcept -> double { return value * (1.0 - value); }

  template <util::Numeric data_t, size_t dim_s>
  [[nodiscard]] static auto eval(Vec<data_t, dim_s, ArenaType::CPU> const& vec)
  {
    Vec<data_t, dim_s, ArenaType::CPU> retVal;
    for ( size_t i = 0; i < dim_s; ++i ) retVal[i] = eval(vec[i]);
    return retVal;
  }

  template <util::Numeric data_t, size_t dim_s>
  [[nodiscard]] static auto derivative(Vec<data_t, dim_s, ArenaType::CPU> const& vec)
  {
    Vec<data_t, dim_s, ArenaType::CPU> retVal;
    for ( size_t i = 0; i < dim_s; ++i ) retVal[i] = derivative(vec[i]);
    return retVal;
  }
};

struct ReLu {
  [[nodiscard]] static auto eval(double value) noexcept -> double { return value >= 0 ? value : 0.0; }
  [[nodiscard]] static auto derivative(double value) noexcept -> double { return value >= 0 ? 1.0 : 0.0; }

  template <util::Numeric data_t, size_t dim_s>
  [[nodiscard]] static auto eval(Vec<data_t, dim_s, ArenaType::CPU> const& vec)
  {
    Vec<data_t, dim_s, ArenaType::CPU> retVal;
    for ( size_t i = 0; i < dim_s; ++i ) retVal[i] = eval(vec[i]);
    return retVal;
  }

  template <util::Numeric data_t, size_t dim_s>
  [[nodiscard]] static auto derivative(Vec<data_t, dim_s, ArenaType::CPU> const& vec)
  {
    Vec<data_t, dim_s, ArenaType::CPU> retVal;
    for ( size_t i = 0; i < dim_s; ++i ) retVal[i] = derivative(vec[i]);
    return retVal;
  }
};

struct Tanh {};

}  // namespace cntns