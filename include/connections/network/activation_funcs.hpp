#pragma once

#include <algorithm>
#include <numeric>

#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"
#include "connections/util/concepts/types.hpp"

#ifdef CNTNS_USE_CUDA
#include "connections/linear_algebra/gpu/cuda/activations.cuh"
#endif

namespace cntns {

struct Sigmoid {
  template <size_t dim_s>
  [[nodiscard]] static auto eval(Vec<dim_s, ArenaType::CPU> const& vec)
  {
    Vec<dim_s, ArenaType::CPU> result;
    std::transform(vec.begin(), vec.end(), result.begin(),
                   [](auto value) { return 1.0F / (1.0F + std::exp(-value)); });
    return result;
  }

  template <size_t dim_s>
  [[nodiscard]] static auto derivative(Vec<dim_s, ArenaType::CPU> const& vec)
  {
    Vec<dim_s, ArenaType::CPU> result;
    std::transform(vec.begin(), vec.end(), result.begin(),
                   [](auto value) { return value * (1.0F - value); });
    return result;
  }

#ifdef CNTNS_USE_CUDA
  template <size_t dim_s>
  [[nodiscard]] static auto eval(Vec<dim_s, ArenaType::GPU> const& vec)
  {
    Vec<dim_s, ArenaType::GPU> result;
    logsig_kernel<<<std::ceil(dim_s / 512.0), 512>>>(vec.data(), result.data(),
                                                     dim_s);
    return result;
  }

  template <size_t dim_s>
  [[nodiscard]] static auto derivative(Vec<dim_s, ArenaType::GPU> const& vec)
  {
    Vec<dim_s, ArenaType::GPU> result;
    logsig_derivative_kernel<<<std::ceil(dim_s / 512.0), 512>>>(
        vec.data(), result.data(), dim_s);
    return result;
  }
#endif
};

struct ReLu {
  template <size_t dim_s>
  [[nodiscard]] static auto eval(Vec<dim_s, ArenaType::CPU> const& vec)
  {
    Vec<dim_s, ArenaType::CPU> result;
    std::transform(vec.begin(), vec.end(), result.begin(),
                   [](auto value) { return value >= 0 ? value : 0.0; });
    return result;
  }

  template <size_t dim_s>
  [[nodiscard]] static auto derivative(Vec<dim_s, ArenaType::CPU> const& vec)
  {
    Vec<dim_s, ArenaType::CPU> result;
    std::transform(vec.begin(), vec.end(), result.begin(),
                   [](auto value) { return value >= 0 ? 1.0 : 0.0; });
    return result;
  }

#ifdef CNTNS_USE_CUDA
  template <size_t dim_s>
  [[nodiscard]] static auto eval(Vec<dim_s, ArenaType::GPU> const& vec)
  {
    Vec<dim_s, ArenaType::GPU> result;
    relu_kernel<<<std::ceil(dim_s / 512.0), 512>>>(vec.data(), result.data(),
                                                   dim_s);
    return result;
  }

  template <size_t dim_s>
  [[nodiscard]] static auto derivative(Vec<dim_s, ArenaType::GPU> const& vec)
  {
    Vec<dim_s, ArenaType::GPU> result;
    relu_derivative_kernel<<<std::ceil(dim_s / 512.0), 512>>>(
        vec.data(), result.data(), dim_s);
    return result;
  }
#endif
};

struct SoftMax {
  template <size_t dim_s>
  [[nodiscard]] static auto eval(Vec<dim_s, ArenaType::CPU> const& vec)
  {
    Vec<dim_s, ArenaType::CPU> prob;
    Vec<dim_s, ArenaType::CPU> exp;
    double                     max = *std::max_element(vec.begin(), vec.end());

    for ( size_t i = 0; i < dim_s; ++i ) {
      exp[i] = std::exp(vec[i] - max);
    }

    double expSum = std::accumulate(exp.begin(), exp.end(), 0.0);
    for ( size_t i = 0; i < dim_s; ++i ) {
      prob[i] = exp[i] / expSum;
    }

    return prob;
  }

  template <size_t dim_s>
  [[nodiscard]] static auto derivative(Vec<dim_s, ArenaType::CPU> const& vec)
  {
    return vec;
  }

#ifdef CNTNS_USE_CUDA
  template <size_t dim_s>
  [[nodiscard]] static auto eval(Vec<dim_s, ArenaType::GPU> const& vec)
  {
    Vec<dim_s, ArenaType::GPU> result;
    logsig_kernel<<<std::ceil(dim_s / 512.0), 512>>>(vec.data(), result.data(),
                                                     dim_s);
    return result;
  }

  template <size_t dim_s>
  [[nodiscard]] static auto derivative(Vec<dim_s, ArenaType::GPU> const& vec)
  {
    return vec;
  }
#endif
};

}  // namespace cntns