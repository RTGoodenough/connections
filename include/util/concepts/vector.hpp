#pragma once

#include <concepts>
#include <type_traits>

#include "util/concepts/conversions.hpp"

namespace cntns::util {

template <typename type_t>
concept is_3d = type_t::DIM == 3;

template <typename first_t, typename second_t>
concept is_same_dim = std::convertible_to<typename first_t::value_t, typename second_t::value_t> &&
    (first_t::DIM == second_t::DIM);

template <typename first_t, typename second_t>
concept is_higher_dim = std::convertible_to<typename first_t::value_t, typename second_t::value_t> &&
    (first_t::DIM > second_t::DIM);

template <typename first_t, typename second_t>
concept is_lower_dim = std::convertible_to<typename first_t::value_t, typename second_t::value_t> &&
    (first_t::DIM < second_t::DIM);
}  // namespace cntns::util