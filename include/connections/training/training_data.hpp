#pragma once

#include <cstddef>
#include <vector>

#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"

namespace cntns {

struct TrainingConfig {
  size_t epochs{};
  size_t batchSize{};
  double learningRate{0.01};
};

/**
 * @brief A struct to hold the training/testing data for the neural network
 * 
 */
template <size_t in_size, size_t out_size, ArenaType arena_e>
struct TrainingData {
  std::vector<Vec<double, in_size, arena_e>>  input;
  std::vector<Vec<double, out_size, arena_e>> correct;

  TrainingData(std::vector<Vec<double, in_size, arena_e>>&&  input,
               std::vector<Vec<double, out_size, arena_e>>&& correct)
      : input{std::move(input)}, correct{std::move(correct)}
  {
  }

  static constexpr size_t IN_SIZE = in_size;
  static constexpr size_t OUT_SIZE = out_size;
};

}  // namespace cntns