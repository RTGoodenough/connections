#pragma once

#include <cstddef>
#include <vector>
#include "linear_algebra/vector.hpp"

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
template <size_t in_size, size_t out_size>
struct TrainingData {
  std::vector<Vec<double, in_size>>  input;
  std::vector<Vec<double, out_size>> correct;

  TrainingData(std::vector<Vec<double, in_size>>&& input, std::vector<Vec<double, out_size>>&& correct)
      : input{std::move(input)}, correct{std::move(correct)}
  {
  }

  static constexpr size_t IN_SIZE = in_size;
  static constexpr size_t OUT_SIZE = out_size;
};

}  // namespace cntns