#pragma once

#include "connections/network/arena.hpp"
#include "connections/training/training.hpp"
#include "network.hpp"

namespace cntns {

/**
 * @brief Utility function to create a neural network with the given cost function and layers
 *
 * @tparam input_size
 * @tparam output_size
 * @tparam cost_t
 * @tparam layer_ts
 */
template <ArenaType arena_e, typename cost_t, typename firstlayer_t,
          typename... layer_ts>
inline constexpr auto create_network(cost_t&& /*cost*/, firstlayer_t&& /**/,
                                     layer_ts&&... layers)
{
  if constexpr ( sizeof...(layers) == 0 ) {
    return NeuralNetwork<arena_e, firstlayer_t::IN_SIZE, firstlayer_t::OUT_SIZE,
                         cost_t, firstlayer_t>();
  }
  else {
    using last_t = std::remove_cvref_t<decltype(std::get<sizeof...(layers) - 1>(
        std::forward_as_tuple(layers...)))>;
    return NeuralNetwork<arena_e, firstlayer_t::IN_SIZE, last_t::OUT_SIZE,
                         cost_t, firstlayer_t, layer_ts...>();
  }
}
}  // namespace cntns