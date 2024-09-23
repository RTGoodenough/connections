#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <type_traits>

#include "connections/network/arena.hpp"
#include "connections/util/concepts/conversions.hpp"
#include "connections/util/concepts/types.hpp"
#include "connections/util/concepts/vector.hpp"

#include "connections/util/operator_crtp.hpp"
#include "connections/util/perf.hpp"

namespace cntns {

#define CNTNS_VEC_FUNC [[nodiscard]] CNTNS_INLINE CNTNS_PERF_FUNC constexpr

template <typename value_t, size_t rows, size_t cols, ArenaType arena_e>
class Matrix;

template <util::Numeric data_t, std::size_t dim_s, ArenaType arena_e>
class Vec : public util::Operators<Vec<data_t, dim_s, arena_e>> {
 public:
  using value_t = data_t;
  using array_t = std::array<value_t, dim_s>;
  static constexpr size_t DIM = dim_s;

  constexpr void zero() { std::fill(_values.begin(), _values.end(), 0.0F); }

  CNTNS_VEC_FUNC auto mag() const noexcept -> double;
  CNTNS_VEC_FUNC auto mag_2() const noexcept -> double;

  template <typename other_t>
  CNTNS_VEC_FUNC auto dot(other_t&& other) const noexcept -> double requires util::is_same_dim<Vec, other_t>;

  template <typename other_t>
  CNTNS_VEC_FUNC auto cross(other_t&& other) const noexcept
      -> Vec requires util::is_3d<Vec> && util::is_same_dim<Vec, other_t>;

  template <typename other_t>
  CNTNS_VEC_FUNC auto hadamard(other_t&& other) const noexcept -> Vec requires util::is_same_dim<Vec, other_t>;

  /**
   * @brief Computes the outer product of two vectors
   * 
   * @tparam other_dim 
   * @param other 
   * @return Matrix<other_dim, dim> 
   */
  template <size_t other_dim>
  CNTNS_VEC_FUNC auto outer_product(Vec<value_t, other_dim, arena_e> const& other) const
      -> Matrix<value_t, other_dim, dim_s, arena_e>;

  CNTNS_VEC_FUNC auto at(size_t idx) const noexcept -> value_t const& { return _values[idx]; }

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

  CNTNS_VEC_FUNC auto operator<=>(Vec const&) const noexcept = default;

  CNTNS_VEC_FUNC auto operator[](size_t idx) const noexcept -> value_t const&
  {
    assert(idx < dim_s);
    return _values[idx];
  }
  CNTNS_VEC_FUNC auto operator[](size_t idx) noexcept -> value_t&
  {
    assert(idx < dim_s);
    return _values[idx];
  }

  CNTNS_VEC_FUNC auto operator+(Vec const& other) const noexcept -> Vec
  {
    array_t temp{_values};
    for ( size_t i = 0; i < dim_s; ++i ) temp[i] += other.at(i);
    return Vec{temp};
  }

  CNTNS_VEC_FUNC auto operator-(Vec const& other) const noexcept -> Vec
  {
    array_t temp{_values};
    for ( size_t i = 0; i < dim_s; ++i ) temp[i] -= other.at(i);
    return Vec{temp};
  }

  template <util::Numeric scalar_t>
  CNTNS_VEC_FUNC auto operator*(scalar_t&& scalar) const noexcept -> Vec
  {
    Vec temp{_values};
    for ( size_t i = 0; i < dim_s; ++i ) temp[i] *= std::forward<scalar_t>(scalar);
    return temp;
  }

  template <util::Numeric scalar_t>
  CNTNS_VEC_FUNC auto operator/(scalar_t&& scalar) const noexcept -> Vec
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

using Vec2_dc = Vec<double, 2, ArenaType::CPU>;
using Vec2_fc = Vec<float, 2, ArenaType::CPU>;
using Vec3_dc = Vec<double, 3, ArenaType::CPU>;
using Vec3_fc = Vec<float, 3, ArenaType::CPU>;

using Vec2_dg = Vec<double, 2, ArenaType::GPU>;
using Vec2_fg = Vec<float, 2, ArenaType::GPU>;
using Vec3_dg = Vec<double, 3, ArenaType::GPU>;
using Vec3_fg = Vec<float, 3, ArenaType::GPU>;

// ==========================================================================================
// ================================== IMPLEMENTATION ========================================
// ==========================================================================================

template <util::Numeric value_t, typename vec_t>
constexpr auto operator*(value_t&& other, vec_t&& vec) -> decltype(auto)
{
  return std::forward<vec_t>(vec) * std::forward<value_t>(other);
}

template <util::Numeric value_t, size_t dim_s, ArenaType arena_e>
constexpr auto Vec<value_t, dim_s, arena_e>::mag() const noexcept -> double
{
  double sum = 0.0;
  for ( size_t i = 0; i < dim_s; ++i ) {
    sum += _values[i] * _values[i];
  }
  return std::sqrt(sum);
}

template <util::Numeric value_t, size_t dim_s, ArenaType arena_e>
constexpr auto Vec<value_t, dim_s, arena_e>::mag_2() const noexcept -> double
{
  double sum = 0.0;
  for ( size_t i = 0; i < dim_s; ++i ) {
    sum += _values[i] * _values[i];
  }
  return sum;
}

template <util::Numeric value_t, size_t dim_s, ArenaType arena_e>
template <typename other_t>
constexpr auto Vec<value_t, dim_s, arena_e>::dot(other_t&& other) const noexcept
    -> double requires util::is_same_dim<Vec, other_t>
{
  double sum = 0.0;
  for ( size_t i = 0; i < dim_s; ++i ) {
    sum += (*this)[i] * static_cast<value_t>(other[i]);
  }
  return sum;
}

template <util::Numeric value_t, size_t dim_s, ArenaType arena_e>
template <typename other_t>
constexpr auto Vec<value_t, dim_s, arena_e>::cross(other_t&& other) const noexcept
    -> Vec requires util::is_3d<Vec> && util::is_same_dim<Vec, other_t>
{
  auto const& vec = *this;
  return Vec{(vec[1] * static_cast<value_t>(other[2]) - vec[2] * static_cast<value_t>(other[1])),
             (vec[2] * static_cast<value_t>(other[0]) - vec[0] * static_cast<value_t>(other[2])),
             (vec[0] * static_cast<value_t>(other[1]) - vec[1] * static_cast<value_t>(other[0]))};
}

template <util::Numeric value_t, size_t dim_s, ArenaType arena_e>
template <typename other_t>
constexpr auto Vec<value_t, dim_s, arena_e>::hadamard(other_t&& other) const noexcept
    -> Vec requires util::is_same_dim<Vec, other_t>
{
  Vec<value_t, dim_s, arena_e> temp{*this};
  for ( size_t i = 0; i < dim_s; ++i ) {
    temp[i] *= static_cast<value_t>(other[i]);
  }
  return temp;
}

/**
   * @brief Computes the outer product of two vectors
   * 
   * @tparam other_dim 
   * @param other 
   * @return Matrix<other_dim, dim> 
   */
template <util::Numeric value_t, size_t dim_s, ArenaType arena_e>
template <size_t other_dim>
CNTNS_VEC_FUNC auto Vec<value_t, dim_s, arena_e>::outer_product(Vec<value_t, other_dim, arena_e> const& other) const
    -> Matrix<value_t, other_dim, dim_s, arena_e>
{
  Matrix<value_t, other_dim, dim_s, arena_e> result{};
  for ( size_t k = 0; k < dim_s; ++k ) {
    for ( size_t j = 0; j < other_dim; ++j ) {
      result(j, k) = _values[k] * other[j];
    }
  }
  return result;
}
}  // namespace cntns