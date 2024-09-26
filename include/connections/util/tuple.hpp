#pragma once

#include <tuple>

namespace cntns::util {
/**
 * @brief Utility function that returns the tail of a tuple
 * 
 * @tparam head 
 * @tparam tail 
 * @param tuple 
 * @return std::tuple<tail...> 
 */
template <typename head, typename... tail>
constexpr auto tuple_tail(std::tuple<head, tail...>& tuple)
    -> std::tuple<tail&...>
{
  return std::apply([](auto&&, auto&... args) { return std::tie(args...); },
                    tuple);
}

/**
 * @brief Utility struct that returns the size of a tuple
 * 
 * @tparam tuple_t 
 */
template <typename tuple_t>
// NOLINTNEXTLINE
struct tuple_size {
  // NOLINTNEXTLINE
  static constexpr size_t value =
      std::tuple_size_v<std::remove_cvref_t<tuple_t>>;
};
}  // namespace cntns::util