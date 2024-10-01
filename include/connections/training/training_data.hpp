#pragma once

#include <cstddef>
#include <vector>

#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"

namespace cntns {

struct TrainingConfig {
  size_t epochs{};
  size_t batchSize{};
  double learningRate{};
};

/**
 * @brief A struct to hold the training/testing data for the neural network
 * 
 */
template <size_t in_size, size_t out_size>
struct TrainingData {
  std::vector<Vec<in_size, ArenaType::CPU>>  input{};
  std::vector<Vec<out_size, ArenaType::CPU>> correct{};

  TrainingData(std::vector<Vec<in_size, ArenaType::CPU>>&&  input,
               std::vector<Vec<out_size, ArenaType::CPU>>&& correct)
      : input{std::move(input)}, correct{std::move(correct)}
  {
  }

  static constexpr size_t IN_SIZE = in_size;
  static constexpr size_t OUT_SIZE = out_size;
};

/**
 * @brief A struct to hold the training/testing data for the neural network
 * 
 */
template <size_t in_size, size_t out_size>
struct TestingData {
  std::vector<Vec<in_size, ArenaType::CPU>>  input;
  std::vector<Vec<out_size, ArenaType::CPU>> correct;

  TestingData(std::vector<Vec<in_size, ArenaType::CPU>>&&  input,
              std::vector<Vec<out_size, ArenaType::CPU>>&& correct)
      : input{std::move(input)}, correct{std::move(correct)}
  {
  }

  static constexpr size_t IN_SIZE = in_size;
  static constexpr size_t OUT_SIZE = out_size;
};

/**
 * @brief Copies the loaded MNIST data over to the GPU in one chunk, returning in a pair of vectors
 * 
 */
template <size_t in_size, size_t out_size>
inline auto gpu_load_data(TrainingData<in_size, out_size> const& data)
    -> std::pair<std::vector<Vec<in_size, ArenaType::GPU>>,
                 std::vector<Vec<out_size, ArenaType::GPU>>>
{
  std::vector<Vec<in_size, ArenaType::GPU>>  input{};
  std::vector<Vec<out_size, ArenaType::GPU>> correct{};

  for ( size_t i = 0; i < data.input.size(); ++i ) {
    input.emplace_back(data.input[i]);
    correct.emplace_back(data.correct[i]);
  }

  return std::pair{std::move(input), std::move(correct)};
}
}  // namespace cntns