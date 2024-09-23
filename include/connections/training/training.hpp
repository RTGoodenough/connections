#pragma once

#include <type_traits>

#include "connections/network/arena.hpp"
#include "connections/network/network.hpp"

#include "connections/training/training_data.hpp"
#include "connections/util/output.hpp"

namespace cntns {

template <size_t in_size, size_t out_size, typename... layer_ts>
void gpu_train_network(auto&& /*network*/, TrainingConfig /*config*/,
                       TrainingData<in_size, out_size, ArenaType::GPU>& /*data*/);
template <size_t in_size, size_t out_size, typename... layer_ts>
void cpu_train_network(auto&& /*network*/, TrainingConfig /*config*/,
                       TrainingData<in_size, out_size, ArenaType::CPU>& /*data*/);

/**
 * @brief Calls the correct training function based on the arena selected
 * 
 */
template <ArenaType arena_e, typename network_t>
inline void train_network(
    network_t&& network, TrainingConfig config,
    TrainingData<std::remove_cvref_t<network_t>::IN_SIZE, std::remove_cvref_t<network_t>::OUT_SIZE, arena_e>& data)
{
  if constexpr ( arena_e == ArenaType::CPU ) {
    cpu_train_network(std::forward<network_t>(network), config, data);
  }
  else {
    gpu_train_network(std::forward<network_t>(network), config, data);
  }
}

/**
 * @brief Trains a network on the GPU
 * 
 */
template <size_t in_size, size_t out_size, typename... layer_ts>
void gpu_train_network(auto&& network, TrainingConfig config, TrainingData<in_size, out_size, ArenaType::GPU>& data)
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
}

/**
 * @brief Trains a network on the CPU
 * 
 */
template <size_t in_size, size_t out_size, typename... layer_ts>
void cpu_train_network(auto&& network, TrainingConfig config, TrainingData<in_size, out_size, ArenaType::CPU>& data)
{
  // Run the training loop for the specified number of epochs
  for ( size_t epoch = 0; epoch < config.epochs; ++epoch ) {
    double loss = 0.0F;

    // Break the data into batches and train the network
    for ( size_t i = 0; i < data.input.size(); i += config.batchSize ) {
      // Evaluate the network for each input in the batch, calculate the loss and back propagate the error
      for ( size_t j = 0; j < config.batchSize; ++j ) {
        auto const& result = network.evaluate(data.input[i + j]);
        loss += network.loss(data.correct[i + j], result);
        network.back_propagate(data.input[i + j], network.error(data.correct[i + j], result));
      }

      // Update the weights of the network with the average loss
      network.update_weights(config.learningRate, config.batchSize);

      output::message("CPU Training\nEpoch: %zu/%zu\nImage: %zu/%zu\nLoss: %f\n", epoch, config.epochs, i,
                      data.input.size(), loss / static_cast<double>(i + 1));
    }
  }
}
}  // namespace cntns