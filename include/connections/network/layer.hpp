#pragma once

#include <cstddef>

#include "connections/linear_algebra/matrix.hpp"
#include "connections/linear_algebra/vector.hpp"

#include "connections/network/arena.hpp"

namespace cntns {
template <size_t input_s, size_t output_s, typename activation_t,
          ArenaType arena_e>
class Layer {
 public:
  static constexpr size_t IN_SIZE = input_s;
  static constexpr size_t OUT_SIZE = output_s;
  using in_vec_t = Vec<input_s, arena_e>;
  using out_vec_t = Vec<output_s, arena_e>;
  using weights_t = Matrix<output_s, input_s, arena_e>;

  [[nodiscard]] constexpr auto evaluate(in_vec_t const& input)
      -> out_vec_t const&;
  [[nodiscard]] constexpr auto back_propagate(in_vec_t const& input,
                                              out_vec_t&& error) -> in_vec_t;

  void update_weights(float learningRate, size_t batchSize);

  [[nodiscard]] constexpr auto get_weights() const -> weights_t const&
  {
    return _weights;
  }
  [[nodiscard]] constexpr auto get_biases() const -> out_vec_t const&
  {
    return _biases;
  }
  [[nodiscard]] constexpr auto get_activations() const -> out_vec_t const&
  {
    return _activations;
  }

 private:
  weights_t _weights;
  out_vec_t _biases;
  out_vec_t _activations;

  weights_t _weightDeltas;
  out_vec_t _biasDeltas;

 public:
  Layer() : _weights{weights_t::random()}, _biases{out_vec_t::random()} {}
  constexpr Layer(Layer const&) = default;
  constexpr Layer(Layer&&) noexcept = default;
  constexpr auto operator=(Layer const&) -> Layer& = default;
  constexpr auto operator=(Layer&&) noexcept -> Layer& = default;
  ~Layer() = default;
};

template <size_t input_s, size_t output_s, typename activation_t,
          ArenaType arena_e>
constexpr auto Layer<input_s, output_s, activation_t, arena_e>::evaluate(
    in_vec_t const& input) -> out_vec_t const&
{
  // Transform input by the neuron weights, add bias, and apply the activation function
  // a_l = o_l(w_l * a_l-1 + b_l)

  _activations = activation_t::eval((_weights * input) += _biases);

  return _activations;
}

template <size_t input_s, size_t output_s, typename activation_t,
          ArenaType arena_e>
constexpr auto Layer<input_s, output_s, activation_t, arena_e>::back_propagate(
    in_vec_t const& input, out_vec_t&& error) -> in_vec_t
{
  // Move the error back through the layer

  // First apply the derivative of the activation function
  error *= activation_t::derivative(_activations);

  // Update the weight and bias deltas

  // dC/dW = error * input
  _weightDeltas += input.outer_product(error);

  // dB = error
  _biasDeltas += error;

  // Return the error for the next layer
  return _weights.transpose() * error;
}

/**
 * @brief Updates the weights of the layer
 * 
 */
template <size_t input_s, size_t output_s, typename activation_t,
          ArenaType arena_e>
void Layer<input_s, output_s, activation_t, arena_e>::update_weights(
    float learningRate, size_t batchSize)
{
  float const learningScale = (learningRate / static_cast<float>(batchSize));
  _weights += (_weightDeltas *= learningScale);
  _biases += (_biasDeltas *= learningScale);

  _biasDeltas.zero();
  _weightDeltas.zero();
}

}  // namespace cntns
