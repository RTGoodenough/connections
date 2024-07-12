#pragma once

#include <cstddef>
#include <vector>

namespace cntns {
template <typename data_t, size_t cols, size_t rows>
class Matrix {
 public:
  using value_t = data_t;

 private:
  std::vector<value_t> _values;

 public:
  Matrix() : _values(cols * rows) {}
  ~Matrix() = default;
  Matrix(Matrix const&) = default;
  Matrix(Matrix&&) noexcept = default;
  auto operator=(Matrix const&) -> Matrix& = default;
  auto operator=(Matrix&&) noexcept -> Matrix& = default;
};

using Matrix_3d = Matrix<double, 3, 3>;
using Matrix_3f = Matrix<double, 3, 3>;
using Matrix_2d = Matrix<double, 2, 2>;
using Matrix_2f = Matrix<double, 2, 2>;
}  // namespace cntns