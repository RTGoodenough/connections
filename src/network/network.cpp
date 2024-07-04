

#include "network/network.hpp"

namespace cntns {

void Network::add_layer(Layer&& layer) { _layers.emplace_back(std::move(layer)); }

}  // namespace cntns