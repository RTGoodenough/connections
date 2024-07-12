#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "arena.hpp"

#include "util/tuple.hpp"

namespace cntns {
template <ArenaType arena_e, size_t in_size, size_t out_size, typename... layer_ts>
class NeuralNetwork {
 public:
  static constexpr ArenaType ARENA = arena_e;
  static constexpr size_t    IN_SIZE = in_size;
  static constexpr size_t    OUT_SIZE = out_size;

  /**
   * @brief Runs the input through the neural network, returning the result. Running it through the mapping function if provided
   * 
   */
  [[nodiscard]] constexpr auto evaluate(auto const& input) { return chain_layers(input, _layers); }

  /**
   * @brief Back propagates the error through the neural network
   * 
   */
  void back_propagate(auto const& input, auto&& error)
  {
    back_propagate_layers(input, std::forward<decltype(error)>(error), _layers);
  }

  /**
   * @brief Updates the weights of the neural network
   * 
   */
  void update_weights(float learningRate, size_t batchSize)
  {
    std::apply([learningRate, batchSize](auto&&... layer) { (layer.update_weights(learningRate, batchSize), ...); },
               _layers);
  }

 private:
  std::tuple<layer_ts...> _layers;

  /**
   * @brief Chains the output of each layer together to get the final result
   * 
   */
  template <typename result_t, typename tail_t>
  [[nodiscard]] constexpr auto chain_layers(result_t&& prevResult, tail_t&& tail) -> decltype(auto)
  {
    // Recursively evaluate each layer, passing the result of the previous layer as input to the next
    if constexpr ( util::tuple_size<tail_t>::value > 1 ) {
      return chain_layers(std::get<0>(tail).evaluate(prevResult), tuple_tail(tail));
    }
    else {
      // Have reached the last layer return the total result, static check that it matches the output
      static_assert(std::remove_cvref_t<decltype(std::get<0>(tail))>::OUT_SIZE == out_size,
                    "Output size of last layer does not match output size of neural network");
      return std::get<0>(tail).evaluate(prevResult);
    }
  }

  /**
   * @brief Back propagates the error through each layer.
   * 
   */
  template <typename input_t, typename error_t, typename tail_t>
  constexpr auto back_propagate_layers(input_t&& input, error_t&& finalError, tail_t&& tail) -> decltype(auto)
  {
    // Recursively back propagate the error through each layer, passing the error of the next layer up the chain
    if constexpr ( util::tuple_size<tail_t>::value == 0 ) {
      // Reached last layer, return the final error to bubble up
      return std::forward<error_t>(finalError);
    }
    else {
      // Bottom up, recursively drill down then back_propagate back up
      return std::get<0>(tail).back_propagate(
          input, back_propagate_layers(std::get<0>(tail).get_activations(), std::forward<error_t>(finalError),
                                       tuple_tail(tail)));
    }
  }

 public:
  constexpr explicit NeuralNetwork(layer_ts&&... layers) {}

  template <ArenaType other_arena_e>
  requires(other_arena_e != arena_e) constexpr explicit NeuralNetwork(
      NeuralNetwork<other_arena_e, in_size, out_size, layer_ts...>&& other)
      : _layers(other._layers)
  {
  }

  constexpr NeuralNetwork() = default;
  constexpr NeuralNetwork(NeuralNetwork const&) = default;
  constexpr NeuralNetwork(NeuralNetwork&&) noexcept = default;
  constexpr auto operator=(NeuralNetwork const&) -> NeuralNetwork& = default;
  constexpr auto operator=(NeuralNetwork&&) noexcept -> NeuralNetwork& = default;
  constexpr ~NeuralNetwork() = default;
};
}  // namespace cntns