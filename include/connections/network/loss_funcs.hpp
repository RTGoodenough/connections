#pragma once

#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"
namespace cntns {

struct MSE {
  template <util::Numeric data_t, std::size_t dim_s>
  [[nodiscard]] static auto error(Vec<data_t, dim_s, ArenaType::CPU> const& correct,
                                  Vec<data_t, dim_s, ArenaType::CPU> const& result)
      -> Vec<data_t, dim_s, ArenaType::CPU>
  {
    return result - correct;
  }

  template <util::Numeric data_t, std::size_t dim_s>
  [[nodiscard]] static auto loss(Vec<data_t, dim_s, ArenaType::CPU> const& correct,
                                 Vec<data_t, dim_s, ArenaType::CPU> const& result) -> double
  {
    double lossVal{};

    for ( size_t i = 0; i < dim_s; ++i ) lossVal += (result[i] - correct[i]) * (result[i] - correct[i]);

    return lossVal * 0.5;
  }
};

struct CrossEntropy {};

}  // namespace cntns