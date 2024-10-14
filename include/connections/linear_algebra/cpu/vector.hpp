#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <random>
#include <type_traits>

#include "connections/network/arena.hpp"
#include "connections/util/concepts/conversions.hpp"
#include "connections/util/concepts/types.hpp"
#include "connections/util/concepts/vector.hpp"

#include "connections/util/operator_crtp.hpp"
#include "connections/util/perf.hpp"

namespace cntns {

#define CNTNS_VEC_FUNC [[nodiscard]] CNTNS_INLINE CNTNS_PERF_FUNC constexpr

template <size_t dim_s, ArenaType arena_e>
class Vec;
template <size_t rows, size_t cols, ArenaType arena_e>
class Matrix;

template <size_t dim_s>
class Vec<dim_s, ArenaType::CPU>
    : public util::Operators<Vec<dim_s, ArenaType::CPU>> {
 public:
  using array_t = std::array<float, dim_s>;
  static constexpr size_t DIM = dim_s;

  static auto random() -> Vec;

  constexpr void zero() { std::fill(_values.begin(), _values.end(), 0.0F); }

  CNTNS_VEC_FUNC auto data() const noexcept -> array_t const&
  {
    return _values;
  }
  CNTNS_VEC_FUNC auto data() noexcept -> array_t& { return _values; }

  CNTNS_VEC_FUNC auto mag() const noexcept -> float;
  CNTNS_VEC_FUNC auto mag_2() const noexcept -> float;

  template <typename other_t>
  CNTNS_VEC_FUNC auto dot(other_t&& other) const noexcept
      -> float requires util::is_same_dim<Vec, other_t>;

  template <typename other_t>
  CNTNS_VEC_FUNC auto cross(other_t&& other) const noexcept
      -> Vec requires util::is_3d<Vec> && util::is_same_dim<Vec, other_t>;

  /**
   * @brief Computes the outer product of two vectors
   * 
   * @tparam other_dim 
   * @param other 
   * @return Matrix<other_dim, dim> 
   */
  template <size_t other_dim>
  CNTNS_VEC_FUNC auto outer_product(Vec<other_dim, ArenaType::CPU> const& other)
      const -> Matrix<other_dim, dim_s, ArenaType::CPU>;

  CNTNS_VEC_FUNC auto at(size_t idx) const noexcept -> float const&
  {
    return _values[idx];
  }

  CNTNS_VEC_FUNC auto begin() { return _values.begin(); }
  CNTNS_VEC_FUNC auto end() { return _values.end(); }
  CNTNS_VEC_FUNC auto begin() const { return _values.begin(); }
  CNTNS_VEC_FUNC auto end() const { return _values.end(); }
  CNTNS_VEC_FUNC auto cbegin() const { return _values.cbegin(); }
  CNTNS_VEC_FUNC auto cend() const { return _values.cend(); }

 private:
  array_t _values;

 public:
  constexpr explicit Vec(array_t const& vals) noexcept : _values(vals) {}
  constexpr explicit Vec(array_t&& vals) noexcept : _values(vals) {}

  // template <typename... args_t>
  // requires util::all_convertible_to<args_t...>
  // constexpr explicit Vec(args_t const&... args) noexcept
  //     : _values(to_array(args...))
  // {
  //   static_assert(sizeof...(args) <= dim_s);
  // }

  template <typename other_t>
  constexpr explicit Vec(
      other_t const& other) noexcept requires util::is_lower_dim<other_t, Vec>
  {
    for ( size_t i = 0; i < other_t::DIM; ++i ) {
      _values[i] = other.at(i);
    }
  }

  template <typename other_t>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload) : incorrect error, does not accept same type
  constexpr explicit Vec(
      other_t&& other) noexcept requires util::is_lower_dim<other_t, Vec> &&
      (not std::is_same_v<other_t, Vec>)
  {
    for ( size_t i = 0; i < other_t::DIM; ++i ) {
      _values[i] = static_cast<float>(other[i]);
    }
  }

  template <typename other_t>
  constexpr auto operator=(other_t const& other) noexcept
      -> Vec& requires util::is_lower_dim<other_t, Vec> &&
      (not std::is_same_v<other_t, Vec>)
  {
    for ( size_t i = 0; i < other_t::DIM; ++i ) {
      _values[i] = static_cast<float>(other[i]);
    }
    return *this;
  }

  template <typename other_t>
  constexpr auto operator=(other_t&& other) noexcept
      -> Vec& requires util::is_lower_dim<other_t, Vec> &&
      (not std::is_same_v<other_t, Vec>)
  {
    for ( size_t i = 0; i < other_t::DIM; ++i ) {
      _values[i] = static_cast<float>(other[i]);
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

  CNTNS_VEC_FUNC auto operator[](size_t idx) const noexcept -> float const&
  {
    assert(idx < dim_s);
    return _values[idx];
  }
  CNTNS_VEC_FUNC auto operator[](size_t idx) noexcept -> float&
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

  CNTNS_VEC_FUNC auto operator*(Vec const& other) const noexcept -> Vec
  {
    Vec result{};
    std::transform(_values.begin(), _values.end(), other._values.begin(),
                   result._values.begin(),
                   [](auto left, auto right) { return left * right; });
    return result;
  }

  template <util::Numeric scalar_t>
  CNTNS_VEC_FUNC auto operator*(scalar_t&& scalar) const noexcept -> Vec
  {
    Vec temp{_values};
    for ( size_t i = 0; i < dim_s; ++i )
      temp[i] *= std::forward<scalar_t>(scalar);
    return temp;
  }

  template <util::Numeric scalar_t>
  CNTNS_VEC_FUNC auto operator/(scalar_t&& scalar) const noexcept -> Vec
  {
    Vec temp{_values};
    for ( size_t i = 0; i < dim_s; ++i )
      temp[i] *= std::forward<scalar_t>(scalar);
    return temp;
  }

 private:
  template <typename... args_t>
  constexpr auto to_array(args_t&&... args)
      -> std::array<float, sizeof...(args_t)>
  {
    return {{(static_cast<float>(std::forward<args_t>(args)))...}};
  }
};

using Vec2_c = Vec<2, ArenaType::CPU>;
using Vec2_c = Vec<2, ArenaType::CPU>;
using Vec3_c = Vec<3, ArenaType::CPU>;
using Vec3_c = Vec<3, ArenaType::CPU>;

using Vec2_g = Vec<2, ArenaType::GPU>;
using Vec2_g = Vec<2, ArenaType::GPU>;
using Vec3_g = Vec<3, ArenaType::GPU>;
using Vec3_g = Vec<3, ArenaType::GPU>;

// ==========================================================================================
// ================================== IMPLEMENTATION ========================================
// ==========================================================================================

template <size_t dim_s>
auto Vec<dim_s, ArenaType::CPU>::random() -> Vec
{
  Vec                             result{};
  std::random_device              rand;
  std::default_random_engine      gen(rand());
  std::normal_distribution<float> dist(-1.0F, 1.0F);

  std::generate(result._values.begin(), result._values.end(),
                [&dist, &gen]() { return dist(gen); });
  return result;
}

template <typename vec_t>
constexpr auto operator*(float&& other, vec_t&& vec) -> decltype(auto)
{
  return std::forward<vec_t>(vec) * std::forward<float>(other);
}

template <size_t dim_s>
constexpr auto Vec<dim_s, ArenaType::CPU>::mag() const noexcept -> float
{
  float sum = 0.0;
  for ( size_t i = 0; i < dim_s; ++i ) {
    sum += _values[i] * _values[i];
  }
  return std::sqrt(sum);
}

template <size_t dim_s>
constexpr auto Vec<dim_s, ArenaType::CPU>::mag_2() const noexcept -> float
{
  float sum = 0.0;
  for ( size_t i = 0; i < dim_s; ++i ) {
    sum += _values[i] * _values[i];
  }
  return sum;
}

template <size_t dim_s>
template <typename other_t>
constexpr auto Vec<dim_s, ArenaType::CPU>::dot(other_t&& other) const noexcept
    -> float requires util::is_same_dim<Vec, other_t>
{
  float sum = 0.0;
  for ( size_t i = 0; i < dim_s; ++i ) {
    sum += (*this)[i] * static_cast<float>(other[i]);
  }
  return sum;
}

template <size_t dim_s>
template <typename other_t>
constexpr auto Vec<dim_s, ArenaType::CPU>::cross(other_t&& other) const noexcept
    -> Vec requires util::is_3d<Vec> && util::is_same_dim<Vec, other_t>
{
  auto const& vec = *this;
  return Vec{(vec[1] * static_cast<float>(other[2]) -
              vec[2] * static_cast<float>(other[1])),
             (vec[2] * static_cast<float>(other[0]) -
              vec[0] * static_cast<float>(other[2])),
             (vec[0] * static_cast<float>(other[1]) -
              vec[1] * static_cast<float>(other[0]))};
}

/**
   * @brief Computes the outer product of two vectors
   * 
   * @tparam other_dim 
   * @param other 
   * @return Matrix<other_dim, dim> 
   */
template <size_t dim_s>
template <size_t other_dim>
CNTNS_VEC_FUNC auto Vec<dim_s, ArenaType::CPU>::outer_product(
    Vec<other_dim, ArenaType::CPU> const& other) const
    -> Matrix<other_dim, dim_s, ArenaType::CPU>
{
  Matrix<other_dim, dim_s, ArenaType::CPU> result{};
  for ( size_t k = 0; k < dim_s; ++k ) {
    for ( size_t j = 0; j < other_dim; ++j ) {
      result(j, k) = _values[k] * other[j];
    }
  }
  return result;
}
}  // namespace cntns