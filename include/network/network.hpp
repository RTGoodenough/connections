#pragma once

#include "network/layer.hpp"

namespace cntns {
class Network {
 public:
  void add_layer(Layer&&);

 private:
  std::vector<Layer> _layers;

  Eigen::Index _num_inputs{};
  Eigen::Index _num_outputs{};

 public:
};
}  // namespace cntns