#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <type_traits>

#include "util/concepts/conversions.hpp"
#include "util/concepts/types.hpp"
#include "util/concepts/vector.hpp"

#include "util/operator_crtp.hpp"

namespace cntns {

template <util::Numeric data_t, std::size_t dim_s>
class Vec : public util::Operators<Vec<data_t, dim_s>> {
 public:
  using value_t = data_t;
  using array_t = std::array<value_t, dim_s>;
  static constexpr size_t DIM = dim_s;

  [[nodiscard]] constexpr auto mag() const noexcept -> double;
  [[nodiscard]] constexpr auto mag_2() const noexcept -> double;

  template <typename other_t>
  [[nodiscard]] constexpr auto dot(other_t&& other) const noexcept -> double requires util::is_same_dim<Vec, other_t>;

  template <typename other_t>
  [[nodiscard]] constexpr auto cross(other_t&& other) const noexcept
      -> Vec requires util::is_3d<Vec> && util::is_same_dim<Vec, other_t>;

  template <typename other_t>
  [[nodiscard]] constexpr auto hadamard(other_t&& other) const noexcept -> Vec requires util::is_same_dim<Vec, other_t>;

  [[nodiscard]] constexpr auto at(size_t idx) const noexcept -> value_t const& { return _values[idx]; }

 private:
  array_t _values;

 public:
  constexpr explicit Vec(array_t const& vals) noexcept : _values(vals) {}
  constexpr explicit Vec(array_t&& vals) noexcept : _values(vals) {}

  template <typename... args_t>
  requires util::all_convertible_to<value_t, args_t...>
  constexpr explicit Vec(args_t const&... args) noexcept : _values(to_array(args...))
  {
    static_assert(sizeof...(args) <= dim_s);
  }

  template <typename other_t>
  constexpr explicit Vec(other_t const& other) noexcept requires util::is_lower_dim<other_t, Vec>
  {
    for ( size_t i = 0; i < other_t::DIM; ++i ) {
      _values[i] = other.at(i);
    }
  }

  template <typename other_t>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload) : incorrect error, does not accept same type
  constexpr explicit Vec(other_t&& other) noexcept requires util::is_lower_dim<other_t, Vec> &&
      (not std::is_same_v<other_t, Vec>)
  {
    for ( size_t i = 0; i < other_t::DIM; ++i ) {
      _values[i] = static_cast<value_t>(other[i]);
    }
  }

  template <typename other_t>
  constexpr auto operator=(other_t const& other) noexcept
      -> Vec& requires util::is_lower_dim<other_t, Vec> &&(not std::is_same_v<other_t, Vec>)
  {
    for ( size_t i = 0; i < other_t::DIM; ++i ) {
      _values[i] = static_cast<value_t>(other[i]);
    }
    return *this;
  }

  template <typename other_t>
  constexpr auto operator=(other_t&& other) noexcept
      -> Vec& requires util::is_lower_dim<other_t, Vec> &&(not std::is_same_v<other_t, Vec>)
  {
    for ( size_t i = 0; i < other_t::DIM; ++i ) {
      _values[i] = static_cast<value_t>(other[i]);
    }
    return *this;
  }

  ~Vec() noexcept = default;
  constexpr Vec() noexcept = default;
  constexpr Vec(Vec const&) noexcept = default;
  constexpr Vec(Vec&&) noexcept = default;
  constexpr auto operator=(Vec const&) noexcept -> Vec& = default;
  constexpr auto operator=(Vec&&) noexcept -> Vec& = default;

  [[nodiscard]] constexpr auto operator<=>(Vec const&) const noexcept = default;

  [[nodiscard]] constexpr auto operator[](size_t idx) noexcept -> value_t&
  {
    assert(idx < dim_s);

    return _values[idx];
  }

  [[nodiscard]] constexpr auto operator+(Vec const& other) const noexcept -> Vec
  {
    array_t temp{_values};
    for ( size_t i = 0; i < dim_s; ++i ) temp[i] += other.at(i);
    return Vec{temp};
  }

  [[nodiscard]] constexpr auto operator-(Vec const& other) const noexcept -> Vec
  {
    array_t temp{_values};
    for ( size_t i = 0; i < dim_s; ++i ) temp[i] -= other.at(i);
    return Vec{temp};
  }

  template <util::Numeric scalar_t>
  [[nodiscard]] constexpr auto operator*(scalar_t&& scalar) const noexcept -> Vec
  {
    Vec temp{_values};
    for ( size_t i = 0; i < dim_s; ++i ) temp[i] *= std::forward<scalar_t>(scalar);
    return temp;
  }

  template <util::Numeric scalar_t>
  [[nodiscard]] constexpr auto operator/(scalar_t&& scalar) const noexcept -> Vec
  {
    Vec temp{_values};
    for ( size_t i = 0; i < dim_s; ++i ) temp[i] *= std::forward<scalar_t>(scalar);
    return temp;
  }

 private:
  template <typename... args_t>
  constexpr auto to_array(args_t&&... args) -> std::array<value_t, sizeof...(args_t)>
  {
    return {{(static_cast<value_t>(std::forward<args_t>(args)))...}};
  }
};

using Vec2_d = Vec<double, 2>;
using Vec2_f = Vec<float, 2>;

using Vec3_d = Vec<double, 3>;
using Vec3_f = Vec<float, 3>;

// ==========================================================================================
// ================================== IMPLEMENTATION ========================================
// ==========================================================================================

template <util::Numeric value_t, typename vec_t>
constexpr auto operator*(value_t&& other, vec_t&& vec) -> decltype(auto)
{
  return std::forward<vec_t>(vec) * std::forward<value_t>(other);
}

template <util::Numeric value_t, size_t dim_s>
constexpr auto Vec<value_t, dim_s>::mag() const noexcept -> double
{
  double sum = 0.0;
  for ( size_t i = 0; i < dim_s; ++i ) {
    sum += _values[i] * _values[i];
  }
  return std::sqrt(sum);
}

template <util::Numeric value_t, size_t dim_s>
constexpr auto Vec<value_t, dim_s>::mag_2() const noexcept -> double
{
  double sum = 0.0;
  for ( size_t i = 0; i < dim_s; ++i ) {
    sum += _values[i] * _values[i];
  }
  return sum;
}

template <util::Numeric value_t, size_t dim_s>
template <typename other_t>
constexpr auto Vec<value_t, dim_s>::dot(other_t&& other) const noexcept
    -> double requires util::is_same_dim<Vec, other_t>
{
  double sum = 0.0;
  for ( size_t i = 0; i < dim_s; ++i ) {
    sum += (*this)[i] * static_cast<value_t>(other[i]);
  }
  return sum;
}

template <util::Numeric value_t, size_t dim_s>
template <typename other_t>
constexpr auto Vec<value_t, dim_s>::cross(other_t&& other) const noexcept
    -> Vec requires util::is_3d<Vec> && util::is_same_dim<Vec, other_t>
{
  auto const& vec = *this;
  return Vec{(vec[1] * static_cast<value_t>(other[2]) - vec[2] * static_cast<value_t>(other[1])),
             (vec[2] * static_cast<value_t>(other[0]) - vec[0] * static_cast<value_t>(other[2])),
             (vec[0] * static_cast<value_t>(other[1]) - vec[1] * static_cast<value_t>(other[0]))};
}

template <util::Numeric value_t, size_t dim_s>
template <typename other_t>
constexpr auto Vec<value_t, dim_s>::hadamard(other_t&& other) const noexcept
    -> Vec requires util::is_same_dim<Vec, other_t>
{
  Vec<value_t, dim_s> temp{*this};
  for ( size_t i = 0; i < dim_s; ++i ) {
    temp[i] *= static_cast<value_t>(other[i]);
  }
  return temp;
}
}  // namespace cntns