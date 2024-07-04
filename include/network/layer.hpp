#pragma once

#include "Eigen/Eigen"

namespace cntns {
class Layer {
 public:
  [[nodiscard]] constexpr auto evaluate(Eigen::VectorXd const& input) -> Eigen::VectorXd const&;
  [[nodiscard]] constexpr auto back_propagate(Eigen::VectorXd const& input, Eigen::VectorXd&& error);

  constexpr void update_weights(float learning_rate, Eigen::Index batch_size);

  [[nodiscard]] constexpr auto get_weights() const -> Eigen::MatrixXd const& { return _weights; }
  [[nodiscard]] constexpr auto get_biases() const -> Eigen::VectorXd const& { return _biases; }
  [[nodiscard]] constexpr auto get_outputs() const -> Eigen::VectorXd const& { return _outputs; }

 private:
  Eigen::MatrixXd _weights;
  Eigen::VectorXd _biases;
  Eigen::VectorXd _outputs;

  Eigen::Index _num_inputs{};
  Eigen::Index _num_outputs{};

 public:
  Layer(Eigen::Index num_inputs, Eigen::Index num_outputs)
      : _weights{Eigen::MatrixXd::Random(num_outputs, num_inputs)},
        _biases{Eigen::VectorXd::Random(num_outputs)},
        _outputs{Eigen::VectorXd::Zero(num_outputs)},
        _num_inputs{num_inputs},
        _num_outputs{num_outputs} {}

  Layer() = default;
  Layer(Layer const&) = default;
  Layer(Layer&&) noexcept = default;
  auto operator=(Layer const&) -> Layer& = default;
  auto operator=(Layer&&) noexcept -> Layer& = default;
  ~Layer() = default;
};
}  // namespace cntns
