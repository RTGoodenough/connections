#pragma once

#include <algorithm>
#include <type_traits>

#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"
#include "connections/network/network.hpp"

#include "connections/training/training_data.hpp"
#include "connections/util/output.hpp"

namespace cntns {

template <size_t in_size, size_t out_size, typename... layer_ts>
auto gpu_test_network(auto& /*network*/,
                      TestingData<in_size, out_size, ArenaType::GPU>& /*data*/)
    -> double;
template <size_t in_size, size_t out_size, typename... layer_ts>
auto cpu_test_network(auto& /*network*/,
                      TestingData<in_size, out_size, ArenaType::CPU>& /*data*/)
    -> double;

/**
 * @brief Calls the correct training function based on the arena selected
 * 
 */
template <ArenaType arena_e, typename network_t>
inline auto test_network(
    network_t&                                                      network,
    TestingData<std::remove_cvref_t<network_t>::IN_SIZE,
                std::remove_cvref_t<network_t>::OUT_SIZE, arena_e>& data)
    -> double
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
template <size_t in_size, size_t out_size, typename... layer_ts>
auto gpu_test_network(auto& /*network*/,
                      TestingData<in_size, out_size, ArenaType::GPU>& /*data*/)
    -> double
{
  // #ifndef NO_CUDA
  //   // Load data onto the GPU
  //   auto [input, correct] = gpu_load_data(data);

  //   // Run the training loop for the specified number of epochs
  //   for ( size_t epoch = 0; epoch < config.epochs; ++epoch ) {
  //     GPU::Float loss{};

  //     // Break the data into batches and train the network
  //     for ( size_t i = 0; i < data.input.size(); i += config.batchSize ) {
  //       // Evaluate the network for each input in the batch, calculate the loss and back propagate the error
  //       for ( size_t j = 0; j < config.batchSize; ++j ) {
  //         auto const& result = network.evaluate(input[i + j]);
  //         loss += network.loss(correct[i + j], result);
  //         network.back_propagate(input[i + j], network.error(correct[i + j], result));
  //       }

  //       // Update the weights of the network with the average loss
  //       network.update_weights(config.learningRate, config.batchSize);

  //       Output::message("GPU Training\nEpoch: %zu/%zu\nImage: %zu/%zu\nLoss: %f\nGPU Memory: %f%\n", epoch, config.epochs,
  //                       i, data.input.size(), loss.value() / static_cast<float>(i + 1), NN::GPU::memory_usage());
  //     }
  //   }
  // #else
  //   throw std::runtime_error("CUDA is not enabled, cannot train network on GPU.");
  // #endif
  return 0;
}

/**
 * @brief Tests a network on the CPU
 * 
 */
template <size_t in_size, size_t out_size, typename... layer_ts>
auto cpu_test_network(auto&                                           network,
                      TestingData<in_size, out_size, ArenaType::CPU>& data)
    -> double
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

  return static_cast<double>(correctCount) /
         static_cast<double>(data.input.size());
}
}  // namespace cntns