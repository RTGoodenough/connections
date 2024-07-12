#pragma once

#include <concepts>
#include <type_traits>

namespace cntns::util {
template <typename value_t, typename... args_t>
concept all_same = (std::is_same_v<args_t, value_t>, ...);

template <typename value_t, typename... args_t>
concept all_convertible_to = (std::convertible_to<args_t, value_t> && ...);
}  // namespace cntns::util