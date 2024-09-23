#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <random>
#include <type_traits>
#include <utility>

#include "connections/linear_algebra/vector.hpp"
#include "connections/network/arena.hpp"
#include "connections/util/operator_crtp.hpp"

namespace cntns {

/**
 * @brief Matrix class held on the CPU
 * 
 * @tparam rows 
 * @tparam cols 
 */
template <typename value_t, size_t rows, size_t cols, ArenaType arena_e>
class Matrix : public util::Operators<Matrix<value_t, rows, cols, arena_e>> {
  friend Matrix<value_t, cols, rows, arena_e>;

 public:
  static constexpr size_t NUM_ROWS = rows;
  static constexpr size_t NUM_COLS = cols;

  constexpr void zero() { std::fill(_values.begin(), _values.end(), 0.0F); }

  [[nodiscard]] static auto    random() -> Matrix<value_t, rows, cols, arena_e>;
  [[nodiscard]] constexpr auto transpose() const -> Matrix<value_t, cols, rows, arena_e>;

  [[nodiscard]] constexpr auto values() const -> std::vector<float> const& { return _values; }

  [[nodiscard]] constexpr auto operator()(size_t row, size_t col) -> float& { return _values[row * cols + col]; }

  [[nodiscard]] constexpr auto operator+(Matrix<value_t, rows, cols, arena_e> const& other) const
      -> Matrix<value_t, rows, cols, arena_e>;
  [[nodiscard]] constexpr auto operator-(Matrix<value_t, rows, cols, arena_e> const& other) const
      -> Matrix<value_t, rows, cols, arena_e>;

  [[nodiscard]] constexpr auto operator*(float scalar) const -> Matrix<value_t, rows, cols, arena_e>;
  [[nodiscard]] constexpr auto operator*(Vec<value_t, cols, arena_e> const& input) const -> Vec<value_t, rows, arena_e>;

 private:
  std::vector<float> _values;

 public:
  Matrix() : _values(rows * cols) {}
  constexpr Matrix(Matrix const&) = default;
  constexpr Matrix(Matrix&&) noexcept = default;
  constexpr auto operator=(Matrix const&) -> Matrix& = default;
  constexpr auto operator=(Matrix&&) noexcept -> Matrix& = default;
  constexpr ~Matrix() = default;
};

/**
 * @brief Creats a new matrix with the transpose of the current matrix
 * 
 * @tparam rows 
 * @tparam cols 
 * @return Matrix<value_t, cols, rows, arena_e> 
 */
template <typename value_t, size_t rows, size_t cols, ArenaType arena_e>
[[nodiscard]] constexpr auto Matrix<value_t, rows, cols, arena_e>::transpose() const
    -> Matrix<value_t, cols, rows, arena_e>
{
  Matrix<value_t, cols, rows, arena_e> result{};
  for ( size_t i = 0; i < rows; ++i ) {
    for ( size_t j = 0; j < cols; ++j ) {
      result(j, i) = _values[i * cols + j];
    }
  }
  return result;
}

/**
 * @brief Creates a new matrix with random values
 * 
 * @tparam rows 
 * @tparam cols 
 * @return Matrix<value_t, rows, cols, arena_e> 
 */
template <typename value_t, size_t rows, size_t cols, ArenaType arena_e>
[[nodiscard]] auto Matrix<value_t, rows, cols, arena_e>::random() -> Matrix<value_t, rows, cols, arena_e>
{
  Matrix<value_t, rows, cols, arena_e> result{};
  result._values.resize(rows * cols);
  std::random_device                    rand;
  std::default_random_engine            gen(rand());
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

  for ( auto& value : result._values ) {
    value = dist(gen);
  }

  return result;
}

/**
 * @brief Piecewise addition of two matrices
 * 
 * @tparam rows 
 * @tparam cols 
 * @param other 
 * @return Matrix<value_t, rows, cols, arena_e> 
 */
template <typename value_t, size_t rows, size_t cols, ArenaType arena_e>
constexpr auto Matrix<value_t, rows, cols, arena_e>::operator+(Matrix<value_t, rows, cols, arena_e> const& other) const
    -> Matrix<value_t, rows, cols, arena_e>
{
  Matrix<value_t, rows, cols, arena_e> result{};
  std::transform(_values.begin(), _values.end(), other._values.begin(), result._values.begin(),
                 [](auto left, auto right) { return left + right; });
  return result;
}

/**
 * @brief Piecewise subtraction of two matrices
 * 
 * @tparam rows 
 * @tparam cols 
 * @param other 
 * @return Matrix<value_t, rows, cols, arena_e> 
 */
template <typename value_t, size_t rows, size_t cols, ArenaType arena_e>
constexpr auto Matrix<value_t, rows, cols, arena_e>::operator-(Matrix<value_t, rows, cols, arena_e> const& other) const
    -> Matrix<value_t, rows, cols, arena_e>
{
  Matrix<value_t, rows, cols, arena_e> result{};
  std::transform(_values.begin(), _values.end(), other._values.begin(), result._values.begin(),
                 [](auto left, auto right) { return left - right; });
  return result;
}

/**
 * @brief Matrix-vector multiplication
 * 
 * @tparam rows 
 * @tparam cols 
 * @param input 
 * @return Vec<rows> 
 */
template <typename value_t, size_t rows, size_t cols, ArenaType arena_e>
[[nodiscard]] constexpr auto Matrix<value_t, rows, cols, arena_e>::operator*(
    Vec<value_t, cols, arena_e> const& input) const -> Vec<value_t, rows, arena_e>
{
  Vec<value_t, rows, arena_e> result{};
  for ( size_t i = 0; i < rows; ++i ) {
    for ( size_t j = 0; j < cols; ++j ) {
      result[i] += _values[i * cols + j] * input[j];
    }
  }

  return result;
}

/**
 * @brief Scalar multiplication of a matrix
 * 
 * @tparam rows 
 * @tparam cols 
 * @param scalar 
 * @return Matrix<value_t, rows, cols, arena_e> 
 */
template <typename value_t, size_t rows, size_t cols, ArenaType arena_e>
[[nodiscard]] constexpr auto Matrix<value_t, rows, cols, arena_e>::operator*(float scalar) const
    -> Matrix<value_t, rows, cols, arena_e>
{
  Matrix<value_t, rows, cols, arena_e> result{};
  std::transform(_values.begin(), _values.end(), result._values.begin(),
                 [scalar](auto value) { return value * scalar; });
  return result;
}
}  // namespace cntns