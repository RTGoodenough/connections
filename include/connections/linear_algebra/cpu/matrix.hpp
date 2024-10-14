#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
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
template <size_t rows, size_t cols>
class Matrix<rows, cols, ArenaType::CPU>
    : public util::Operators<Matrix<rows, cols, ArenaType::CPU>> {
  friend Matrix<cols, rows, ArenaType::CPU>;

  using Mat = Matrix<rows, cols, ArenaType::CPU>;
  using MatTranspose = Matrix<cols, rows, ArenaType::CPU>;
  using InputVec = Vec<cols, ArenaType::CPU>;
  using OutputVec = Vec<rows, ArenaType::CPU>;

 public:
  static constexpr size_t NUM_ROWS = rows;
  static constexpr size_t NUM_COLS = cols;

  constexpr void zero() { std::fill(_values.begin(), _values.end(), 0.0F); }

  [[nodiscard]] static auto    random() -> Mat;
  [[nodiscard]] constexpr auto transpose() const -> MatTranspose;

  [[nodiscard]] constexpr auto values() const -> std::vector<float> const&
  {
    return _values;
  }

  [[nodiscard]] constexpr auto operator()(size_t row, size_t col) -> float&
  {
    return _values[row * cols + col];
  }

  [[nodiscard]] constexpr auto operator+(Mat const& other) const -> Mat;
  [[nodiscard]] constexpr auto operator-(Mat const& other) const -> Mat;

  [[nodiscard]] constexpr auto operator*(float scalar) const -> Mat;
  [[nodiscard]] constexpr auto operator*(InputVec const& input) const
      -> OutputVec;

 private:
  std::vector<float> _values;

 public:
  constexpr Matrix() : _values(rows * cols) {}
  constexpr Matrix(Matrix const&) = default;
  constexpr Matrix(Matrix&&) noexcept = default;
  constexpr auto operator=(Matrix const&) -> Matrix& = default;
  constexpr auto operator=(Matrix&&) noexcept -> Matrix& = default;
  ~Matrix() = default;
};

/**
 * @brief Creats a new matrix with the transpose of the current matrix
 * 
 * @tparam rows 
 * @tparam cols 
 * @return MatTranspose 
 */
template <size_t rows, size_t cols>
[[nodiscard]] constexpr auto Matrix<rows, cols, ArenaType::CPU>::transpose()
    const -> MatTranspose
{
  MatTranspose result{};
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
 * @return Mat 
 */
template <size_t rows, size_t cols>
[[nodiscard]] auto Matrix<rows, cols, ArenaType::CPU>::random() -> Mat
{
  Mat result{};
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
 * @return Mat 
 */
template <size_t rows, size_t cols>
constexpr auto Matrix<rows, cols, ArenaType::CPU>::operator+(
    Mat const& other) const -> Mat
{
  Mat result{};
  std::transform(_values.begin(), _values.end(), other._values.begin(),
                 result._values.begin(),
                 [](auto left, auto right) { return left + right; });
  return result;
}

/**
 * @brief Piecewise subtraction of two matrices
 * 
 * @tparam rows 
 * @tparam cols 
 * @param other 
 * @return Mat 
 */
template <size_t rows, size_t cols>
constexpr auto Matrix<rows, cols, ArenaType::CPU>::operator-(
    Mat const& other) const -> Mat
{
  Mat result{};
  std::transform(_values.begin(), _values.end(), other._values.begin(),
                 result._values.begin(),
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
template <size_t rows, size_t cols>
[[nodiscard]] constexpr auto Matrix<rows, cols, ArenaType::CPU>::operator*(
    InputVec const& input) const -> OutputVec
{
  OutputVec result{};
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
 * @return Mat 
 */
template <size_t rows, size_t cols>
[[nodiscard]] constexpr auto Matrix<rows, cols, ArenaType::CPU>::operator*(
    float scalar) const -> Mat
{
  Mat result{};
  std::transform(_values.begin(), _values.end(), result._values.begin(),
                 [scalar](auto value) { return value * scalar; });
  return result;
}
}  // namespace cntns