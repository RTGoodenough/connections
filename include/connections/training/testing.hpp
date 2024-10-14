#pragma once

#include <algorithm>
#include <iostream>
#include <type_traits>

#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"
#include "connections/network/network.hpp"

#include "connections/training/training_data.hpp"
#include "connections/util/gpu.hpp"
#include "connections/util/output.hpp"

namespace cntns {

template <size_t in_size, size_t out_size>
auto gpu_test_network(auto& network, TestingData<in_size, out_size>& data)
    -> float;

template <size_t in_size, size_t out_size>
auto cpu_test_network(auto& /*network*/,
                      TestingData<in_size, out_size>& /*data*/) -> float;

/**
 * @brief Calls the correct training function based on the arena selected
 *
 */
template <ArenaType arena_e, typename network_t>
inline auto test_network(
    network_t&                                             network,
    TestingData<std::remove_cvref_t<network_t>::IN_SIZE,
                std::remove_cvref_t<network_t>::OUT_SIZE>& data) -> float
{
  if constexpr ( arena_e == ArenaType::CPU ) {
    return cpu_test_network(network, data);
  }
  else {
    return gpu_test_network(network, data);
  }
}

/**
 * @brief Trains a network on the GPU
 *
 */
template <size_t in_size, size_t out_size>
auto gpu_test_network(auto& network, TestingData<in_size, out_size>& data)
    -> float
{
#ifdef CNTNS_USE_CUDA
  constexpr size_t FREQ = 100;
  size_t           correctCount = 0;

  for ( size_t i = 0; i < data.input.size(); ++i ) {
    auto result = network.evaluate(Vec<in_size, ArenaType::GPU>(data.input[i]));

    size_t correctIdx = std::distance(
        data.correct[i].begin(),
        std::max_element(data.correct[i].begin(), data.correct[i].end()));

    Vec<out_size, ArenaType::CPU> cpuResult = result.pull();
    int                           resultIdx =
        std::distance(cpuResult.begin(),
                      std::max_element(cpuResult.begin(), cpuResult.end()));

    if ( resultIdx == correctIdx ) {
      ++correctCount;
    }

    if ( i % FREQ == 0 ) {
      output::message("Testing\nImage: %zu/%zu\nCorrect: %zu/%zu\n", i,
                      data.input.size(), correctCount, i + 1);
    }
  }

  output::message("Testing\nImage: %zu/%zu\nCorrect: %zu/%zu\n",
                  data.input.size(), data.input.size(), correctCount,
                  data.input.size());

  return static_cast<float>(correctCount) /
         static_cast<float>(data.input.size());
#else
  throw std::runtime_error("CUDA is not enabled, cannot train network on GPU.");
#endif
}

/**
 * @brief Tests a network on the CPU
 *
 */
template <size_t in_size, size_t out_size>
auto cpu_test_network(auto& network, TestingData<in_size, out_size>& data)
    -> float
{
  constexpr size_t FREQ = 100;

  std::string temp;

  size_t correctCount = 0;
  for ( size_t i = 0; i < data.input.size(); ++i ) {
    auto result = network.evaluate(data.input[i]);

    size_t correctIdx = std::distance(
        data.correct[i].begin(),
        std::max_element(data.correct[i].begin(), data.correct[i].end()));
    size_t resultIdx = std::distance(
        result.begin(), std::max_element(result.begin(), result.end()));

    if ( resultIdx == correctIdx ) {
      ++correctCount;
    }

    if ( i % FREQ == 0 ) {
      output::message("Testing\nImage: %zu/%zu\nCorrect: %zu/%zu\n", i,
                      data.input.size(), correctCount, i + 1);
    }
  }

  output::message("Testing\nImage: %zu/%zu\nCorrect: %zu/%zu\n",
                  data.input.size(), data.input.size(), correctCount,
                  data.input.size());

  return static_cast<float>(correctCount) /
         static_cast<float>(data.input.size());
}
}  // namespace cntns