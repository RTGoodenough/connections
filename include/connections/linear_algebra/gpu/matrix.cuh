#pragma once

#ifdef CNTNS_USE_CUDA

#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "vector.cuh"
#include "cuda/matrix.cuh"

#include "connections/linear_algebra/proxies/transpose_proxy.hpp"

#include "connections/network/arena.hpp"
#include "connections/util/concepts/types.hpp"
#include "connections/util/operator_crtp.hpp"

namespace cntns {

/**
 * @brief Matrix class held on the GPU
 * 
 * @tparam rows 
 * @tparam cols 
 */
template <size_t rows, size_t cols>
class Matrix<rows, cols, ArenaType::GPU> : public util::Operators<Matrix<rows, cols, ArenaType::GPU>> {
 public:
  constexpr static size_t NUM_ROWS = rows;
  constexpr static size_t NUM_COLS = cols;

  constexpr static dim3 BLOCK_SIZE{32, 32, 1};
  constexpr static dim3 GRID_SIZE{static_cast<unsigned int>(std::ceil(cols / static_cast<double>(BLOCK_SIZE.x))),
                                  static_cast<unsigned int>(std::ceil(rows / static_cast<double>(BLOCK_SIZE.y))), 1};

  [[nodiscard]] static auto random() -> Matrix;

  void               zero() { mat_fill_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(data(), 0.0F, rows, cols); }
  [[nodiscard]] auto transpose() const -> TransposeProxy<rows, cols>;

  [[nodiscard]] auto values() const -> Vec<rows * cols, ArenaType::GPU> const& { return _values; }
  [[nodiscard]] auto data() -> double* { return _values.data(); }
  [[nodiscard]] auto data() const -> double const* { return _values.data(); }

  [[nodiscard]] auto pull() const -> std::vector<double>;
  void               push(std::vector<double> const& values);

  [[nodiscard]] auto operator+(Matrix<rows, cols, ArenaType::GPU> const& other) const -> Matrix<rows, cols, ArenaType::GPU>;
  [[nodiscard]] auto operator-(Matrix<rows, cols, ArenaType::GPU> const& other) const -> Matrix<rows, cols, ArenaType::GPU>;
  [[nodiscard]] auto operator*(double scalar) const -> Matrix<rows, cols, ArenaType::GPU>;
  [[nodiscard]] auto operator*(Vec<cols, ArenaType::GPU> const& input) const -> Vec<rows, ArenaType::GPU>;

  auto operator+=(Matrix<rows, cols, ArenaType::GPU> const& other) -> Matrix<rows, cols, ArenaType::GPU>&;
  auto operator+=(OuterProductProxy<rows, cols> const& other) -> Matrix<rows, cols, ArenaType::GPU>&;
  auto operator-=(Matrix<rows, cols, ArenaType::GPU> const& other) -> Matrix<rows, cols, ArenaType::GPU>&;
  auto               operator*=(float scalar) -> Matrix<rows, cols, ArenaType::GPU>&;

 private:
  Vec<rows * cols, ArenaType::GPU> _values{};

 public:
  Matrix() = default;
  ~Matrix() = default;
  Matrix(Matrix&& other) noexcept = default;
  auto operator=(Matrix&& other) noexcept -> Matrix& = default;

  Matrix(Matrix const&) = delete;
  auto operator=(Matrix const&) -> Matrix& = delete;
};

/**
 * @brief Piecewise addition of two matrices
 * 
 * @tparam rows 
 * @tparam cols 
 * @param other 
 * @return Matrix<rows, cols, ArenaType::GPU> 
 */
template <size_t rows, size_t cols>
[[nodiscard]] auto Matrix<rows, cols, ArenaType::GPU>::operator+(Matrix<rows, cols, ArenaType::GPU> const& other) const -> Matrix<rows, cols, ArenaType::GPU> {
  Matrix<rows, cols, ArenaType::GPU> result{};
  mat_add_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_values.data(), other.data(), result.data(), rows, cols);
  util::check_error(cudaGetLastError());
  return result;
}

/**
 * @brief Piecewise subtraction of two matrices
 * 
 * @tparam rows 
 * @tparam cols 
 * @param other 
 * @return Matrix<rows, cols, ArenaType::GPU> 
 */
template <size_t rows, size_t cols>
[[nodiscard]] auto Matrix<rows, cols, ArenaType::GPU>::operator-(Matrix<rows, cols, ArenaType::GPU> const& other) const -> Matrix<rows, cols, ArenaType::GPU> {
  Matrix<rows, cols, ArenaType::GPU> result{};
  mat_sub_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_values.data(), other.data(), result.data(), rows, cols);
  util::check_error(cudaGetLastError());
  return result;
}

/**
 * @brief Scalar multiplication of a matrix
 * 
 * @tparam rows 
 * @tparam cols 
 * @param scalar 
 * @return Matrix<rows, cols, ArenaType::GPU> 
 */
template <size_t rows, size_t cols>
[[nodiscard]] auto Matrix<rows, cols, ArenaType::GPU>::operator*(double scalar) const -> Matrix<rows, cols, ArenaType::GPU> {
  Matrix<rows, cols, ArenaType::GPU> result{};
  mat_scalar_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_values.data(), scalar, result.data(), rows, cols);
  util::check_error(cudaGetLastError());
  return result;
}

/**
 * @brief Matrix-vector multiplication
 * 
 * @tparam rows 
 * @tparam cols 
 * @param input 
 * @return Vec<rows, ArenaType::GPU> 
 */
template <size_t rows, size_t cols>
[[nodiscard]] auto Matrix<rows, cols, ArenaType::GPU>::operator*(Vec<cols, ArenaType::GPU> const& input) const -> Vec<rows, ArenaType::GPU> {
  static dim3 blockSize{512};
  static dim3 gridSize{static_cast<unsigned int>(std::ceil(rows / static_cast<double>(blockSize.x)))};

  Vec<rows, ArenaType::GPU> result{};
  mat_vec_mul_kernel<<<gridSize, blockSize>>>(data(), input.data(), result.data(), rows, cols);
  util::check_error(cudaGetLastError());
  return result;
}

/**
 * @brief Piecewise addition of two matrices in place
 * 
 * @tparam rows 
 * @tparam cols 
 * @param other 
 * @return Matrix<rows, cols, ArenaType::GPU>& 
 */
template <size_t rows, size_t cols>
auto Matrix<rows, cols, ArenaType::GPU>::operator+=(Matrix<rows, cols, ArenaType::GPU> const& other) -> Matrix<rows, cols, ArenaType::GPU>& {
  mat_add_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_values.data(), other.data(), _values.data(), rows, cols);
  util::check_error(cudaGetLastError());
  return *this;
}

/**
 * @brief Piecewise addition of an outer product proxy to a matrix in place
 * 
 * @tparam rows 
 * @tparam cols 
 * @param other 
 * @return Matrix<rows, cols, ArenaType::GPU>& 
 */
template <size_t rows, size_t cols>
auto Matrix<rows, cols, ArenaType::GPU>::operator+=(OuterProductProxy<rows, cols> const& other) -> Matrix<rows, cols, ArenaType::GPU>& {
  mat_outer_product_add_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_values.data(), other.rhs, other.lhs, rows, cols);
  util::check_error(cudaGetLastError());
  return *this;
}

/**
 * @brief Piecewise subtraction of two matrices in place
 * 
 * @tparam rows 
 * @tparam cols 
 * @param other 
 * @return Matrix<rows, cols, ArenaType::GPU>& 
 */
template <size_t rows, size_t cols>
auto Matrix<rows, cols, ArenaType::GPU>::operator-=(Matrix<rows, cols, ArenaType::GPU> const& other) -> Matrix<rows, cols, ArenaType::GPU>& {
  mat_sub_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_values.data(), other.data(), _values.data(), rows, cols);
  util::check_error(cudaGetLastError());
  return *this;
}

/**
 * @brief Scalar multiplication of a matrix in place
 * 
 * @tparam rows 
 * @tparam cols 
 * @param scalar 
 * @return Matrix<rows, cols, ArenaType::GPU>& 
 */
template <size_t rows, size_t cols>
[[nodiscard]] auto Matrix<rows, cols, ArenaType::GPU>::operator*=(float scalar) -> Matrix<rows, cols, ArenaType::GPU>& {
  mat_scalar_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_values.data(), scalar, _values.data(), rows, cols);
  util::check_error(cudaGetLastError());
  return *this;
}


/**
 * @brief Transpose the matrix and return a proxy
 * 
 * @tparam rows 
 * @tparam cols 
 * @return TransposeProxy<rows, cols, ArenaType::GPU> 
 */
template <size_t rows, size_t cols>
[[nodiscard]] auto Matrix<rows, cols, ArenaType::GPU>::transpose() const -> TransposeProxy<rows, cols> {
  return {data()};
}

/**
 * @brief Push a vector of values to the matrix on the GPU
 * 
 * @tparam rows 
 * @tparam cols 
 * @param values 
 */
template <size_t rows, size_t cols>
void Matrix<rows, cols, ArenaType::GPU>::push(std::vector<double> const& values) {
  assert(values.size() == rows * cols);
  _values.push(values);
}

/**
 * @brief Pull the matrix to a vector of values on the CPU
 * 
 * @tparam rows 
 * @tparam cols 
 * @return std::vector<double> 
 */
template <size_t rows, size_t cols>
[[nodiscard]] auto Matrix<rows, cols, ArenaType::GPU>::pull() const -> std::vector<double> {
  std::vector<double> result(rows * cols);
  util::check_error(cudaMemcpy(result.data(), _values.data(), rows * cols * sizeof(double), cudaMemcpyDeviceToHost));
  return result;
}

/**
 * @brief Generate a random matrix
 * 
 * @tparam rows 
 * @tparam cols 
 * @return Matrix 
 */
template <size_t rows, size_t cols>
[[nodiscard]] auto Matrix<rows, cols, ArenaType::GPU>::random() -> Matrix {
  Matrix result{};
  mat_randomize_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(result.data(), -1.0F, 1.0F, rows, cols);
  util::check_error(cudaGetLastError());
  return result;
}
}  // namespace cntns

#endif
