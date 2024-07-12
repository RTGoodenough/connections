#pragma once

#include <cstddef>

#include "linear_algebra/matrix.hpp"
#include "linear_algebra/vector.hpp"

namespace cntns::cpu {
template <size_t input_s, size_t output_s, typename activation_t>
class Layer {
 public:
  static constexpr size_t IN_SIZE = input_s;
  static constexpr size_t OUT_SIZE = output_s;
  using inputVec_t = Vec<double, input_s>;
  using outputVec_t = Vec<double, output_s>;
  using weights_t = Matrix<double, output_s, input_s>;

  [[nodiscard]] constexpr auto evaluate(inputVec_t const& input) -> outputVec_t const&;
  [[nodiscard]] constexpr auto back_propagate(inputVec_t const& input, outputVec_t&& error) -> inputVec_t;

  void update_weights(float learningRate, size_t batchSize);

  [[nodiscard]] constexpr auto get_weights() const -> weights_t const& { return _weights; }
  [[nodiscard]] constexpr auto get_biases() const -> outputVec_t const& { return _biases; }
  [[nodiscard]] constexpr auto get_outputs() const -> outputVec_t const& { return _outputs; }

 private:
  Matrix<double, output_s, input_s> _weights;
  Vec<double, output_s>             _biases;
  Vec<double, output_s>             _outputs;

  Matrix<double, output_s, input_s> _weightDeltas;
  Vec<double, output_s>             _biasDeltas;

 public:
  Layer() = default;
  Layer(Layer const&) = default;
  Layer(Layer&&) noexcept = default;
  auto operator=(Layer const&) -> Layer& = default;
  auto operator=(Layer&&) noexcept -> Layer& = default;
  ~Layer() = default;
};

template <size_t input_s, size_t output_s, typename activation_t>
constexpr auto Layer<input_s, output_s, activation_t>::evaluate(inputVec_t const& input) -> outputVec_t const&
{
  // Transform input by the neuron weights, add bias, and apply the activation function
  // a_l = o_l(w_l * a_l-1 + b_l)

  _outputs = activation_t::eval((_weights * input) += _biases);

  return _outputs;
}

template <size_t input_s, size_t output_s, typename activation_t>
constexpr auto Layer<input_s, output_s, activation_t>::back_propagate(inputVec_t const& input, outputVec_t&& error)
    -> inputVec_t
{
  // Move the error back through the layer

  // First apply the derivative of the activation function
  error *= activation_t::derivative(_outputs);

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
template <size_t input_s, size_t output_s, typename activation_t>
void Layer<input_s, output_s, activation_t>::update_weights(float learningRate, size_t batchSize)
{
  float const learningScale = (learningRate / static_cast<float>(batchSize));
  _weights += (_weightDeltas *= learningScale);
  _biases += (_biasDeltas *= learningScale);

  _biasDeltas.reset();
  _weightDeltas.reset();
}

}  // namespace cntns::cpu
