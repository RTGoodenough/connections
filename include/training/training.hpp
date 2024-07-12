#pragma once

#include "network/network.hpp"

#include "training/training_data.hpp"
#include "util/output.hpp"

namespace cntns {

template <size_t in_size, size_t out_size, typename... layer_ts>
void gpu_train_network(NeuralNetwork<ArenaType::GPU, in_size, out_size, layer_ts...>& /*network*/,
                       TrainingConfig /*config*/, TrainingData<in_size, out_size>& /*data*/);
template <size_t in_size, size_t out_size, typename... layer_ts>
void cpu_train_network(NeuralNetwork<ArenaType::CPU, in_size, out_size, layer_ts...>& /*network*/,
                       TrainingConfig /*config*/, TrainingData<in_size, out_size>& /*data*/);

/**
 * @brief Calls the correct training function based on the arena selected
 * 
 */
template <ArenaType arena_t, typename network_t>
inline void train_network(network_t& network, TrainingConfig config,
                          TrainingData<network_t::IN_SIZE, network_t::OUT_SIZE>& data)
{
  if constexpr ( arena_t == ArenaType::CPU ) {
    cpu_train_network(network, config, data);
  }
  else {
    gpu_train_network(network, config, data);
  }
}

/**
 * @brief Trains a network on the GPU
 * 
 */
template <size_t in_size, size_t out_size, typename... layer_ts>
void gpu_train_network(NeuralNetwork<ArenaType::GPU, in_size, out_size, layer_ts...>& network, TrainingConfig config,
                       TrainingData<in_size, out_size>& data)
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
void cpu_train_network(NeuralNetwork<ArenaType::CPU, in_size, out_size, layer_ts...>& network, TrainingConfig config,
                       TrainingData<in_size, out_size>& data)
{
  // Run the training loop for the specified number of epochs
  for ( size_t epoch = 0; epoch < config.epochs; ++epoch ) {
    float loss = 0.0F;

    // Break the data into batches and train the network
    for ( size_t i = 0; i < data.input.size(); i += config.batchSize ) {
      // Evaluate the network for each input in the batch, calculate the loss and back propagate the error
      for ( size_t j = 0; j < config.batchSize; ++j ) {
        auto const& result = network.evaluate(data.input[i + j]);
        // loss += network.loss(data.correct[i + j], result);
        // network.back_propagate(data.input[i + j], network.error(data.correct[i + j], result));
      }

      // Update the weights of the network with the average loss
      network.update_weights(config.learningRate, config.batchSize);

      output::message("CPU Training\nEpoch: %zu/%zu\nImage: %zu/%zu\nLoss: %f\n", epoch, config.epochs, i,
                      data.input.size(), loss / static_cast<float>(i + 1));
    }
  }
}
}  // namespace cntns