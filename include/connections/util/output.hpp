#pragma once

#include <cstdio>
#include <utility>

namespace cntns::output {
/**
 * @brief Clears the console
 * 
 */
inline void clear_screen()
{
  std::printf("\033[2J");
  std::printf("\033[H");
}

/**
 * @brief Prints a message to the console
 * 
 */
template <typename... arg_ts>
inline void message(char const* str, arg_ts&&... values)
{
  clear_screen();
  if constexpr ( sizeof...(values) == 0 ) {
    std::printf("%s", str);
  }
  else {
    std::printf(str, std::forward<arg_ts>(values)...);
  }
}
}  // namespace cntns::output