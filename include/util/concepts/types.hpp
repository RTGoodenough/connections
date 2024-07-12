#pragma once

#include <type_traits>
namespace cntns::util {

template <typename type_t>
concept Numeric = std::is_arithmetic<std::remove_cvref_t<type_t>>::value;

}