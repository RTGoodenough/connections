#pragma once

#ifdef CNTNS_USE_CUDA

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <cuda_runtime_api.h>

#include "connections/linear_algebra/proxies/outer_product_proxy.hpp"
#include "connections/network/arena.hpp"

#include "connections/util/concepts/types.hpp"
#include "connections/util/gpu.hpp"
#include "connections/util/operator_crtp.hpp"

#include "connections/linear_algebra/gpu/cuda/vector.cuh"

namespace cntns {
template <util::Numeric data_t, size_t dim_s, ArenaType arena_e>
class Vec;

template <typename value_t, size_t rows, size_t cols, ArenaType arena_e>
class Matrix;

template <util::Numeric data_t, size_t dim_s>
class Vec<data_t, dim_s, ArenaType::GPU>
    : public util::Operators<Vec<data_t, dim_s, ArenaType::GPU>> {
  using cVec = Vec<data_t, dim_s, ArenaType::CPU>;

 public:
  static constexpr size_t SIZE = dim_s;
  static constexpr dim3   BLOCK_SIZE{128};
  static constexpr dim3   GRID_SIZE{static_cast<unsigned int>(
      dim_s / static_cast<double>(BLOCK_SIZE.x))};

  [[nodiscard]] static auto random() -> Vec;

  void reset();

  [[nodiscard]] auto data() -> data_t* { return _data; }
  [[nodiscard]] auto data() const -> data_t const* { return _data; }

  [[nodiscard]] auto pull() const -> cVec;
  void               pull(cVec& vec) const;
  void               pull(std::array<data_t, dim_s>& vec) const;
  void               pull(std::vector<data_t>& vec) const;

  void push(cVec const& vec);
  void push(cVec&& vec);
  void push(std::array<data_t, dim_s> const& vec);
  void push(std::array<data_t, dim_s>&& vec);
  void push(std::vector<data_t> const& vec);
  void push(std::vector<data_t>&& vec);

  template <size_t other_dim>
  [[nodiscard]] auto outer_product(
      Vec<data_t, other_dim, ArenaType::GPU> const& other) const
      -> OuterProductProxy<other_dim, dim_s>
  {
    return OuterProductProxy<other_dim, dim_s>{data(), other.data()};
  }

  [[nodiscard]] constexpr auto operator+(Vec const& other) const -> Vec;
  constexpr auto               operator+=(Vec const& other) -> Vec&;

  [[nodiscard]] constexpr auto operator-(Vec const& other) const -> Vec;
  constexpr auto               operator-=(Vec const& other) -> Vec&;

  [[nodiscard]] constexpr auto operator*(Vec const& other) const -> Vec;
  constexpr auto               operator*=(Vec const& other) -> Vec&;

  [[nodiscard]] constexpr auto operator*(data_t scalar) const -> Vec;
  constexpr auto               operator*=(data_t scalar) -> Vec&;

 private:
  data_t* _data{nullptr};

  void allocate()
  {
    if ( _data == nullptr ) {
      util::check_error(
          cudaMallocAsync(&_data, dim_s * sizeof(data_t), nullptr));
    }
  }

 public:
  Vec();
  ~Vec();
  Vec(const Vec& /*vec*/);
  Vec(Vec&& /*vec*/) noexcept;
  auto operator=(const Vec& /*other*/) -> Vec&;
  auto operator=(Vec&& /*other*/) noexcept -> Vec&;

  explicit Vec(Vec<data_t, dim_s, ArenaType::CPU> const& /*vec*/);
  explicit Vec(Vec<data_t, dim_s, ArenaType::CPU>&& /*vec*/);
};

// -------------------------------------------------------------------
// ------------------------- Implementation -------------------------
// -------------------------------------------------------------------

/**
 * @brief Creates an array of random values
 * 
 * @tparam dim_s 
 * @return Array 
 */
template <util::Numeric data_t, size_t dim_s>
auto Vec<data_t, dim_s, ArenaType::GPU>::random()
    -> Vec<data_t, dim_s, ArenaType::GPU>
{
  Vec<data_t, dim_s, ArenaType::GPU> result{};
  vector_randomize_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(result.data(), -1.0F, 1.0F,
                                                     dim_s);
  util::check_error(cudaGetLastError());
  return result;
}

/**
 * @brief Sets all values in the array to 0
 * 
 * @tparam dim_s 
 */
template <util::Numeric data_t, size_t dim_s>
void Vec<data_t, dim_s, ArenaType::GPU>::reset()
{
  vector_reset_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_data, dim_s);
  util::check_error(cudaGetLastError());
}

/**
 * @brief Pulls the data from the gpu into a new Vec
*/
template <util::Numeric data_t, size_t dim_s>
auto Vec<data_t, dim_s, ArenaType::GPU>::pull() const
    -> Vec<data_t, dim_s, ArenaType::CPU>
{
  assert(_data != nullptr);
  Vec<data_t, dim_s, ArenaType::GPU> result{};
  util::check_error(cudaMemcpy(result.data(), _data, dim_s * sizeof(data_t),
                               cudaMemcpyDeviceToHost));
  return result;
}

/**
 * @brief Pulls the data from the gpu into the provided Vec
*/
template <util::Numeric data_t, size_t dim_s>
void Vec<data_t, dim_s, ArenaType::GPU>::pull(
    Vec<data_t, dim_s, ArenaType::CPU>& vec) const
{
  assert(_data != nullptr);
  util::check_error(cudaMemcpy(vec.data(), _data, dim_s * sizeof(data_t),
                               cudaMemcpyDeviceToHost));
}

/**
 * @brief Pulls the data from the gpu into the provided std::array
*/
template <util::Numeric data_t, size_t dim_s>
void Vec<data_t, dim_s, ArenaType::GPU>::pull(
    std::array<data_t, dim_s>& vec) const
{
  assert(_data != nullptr);
  util::check_error(cudaMemcpy(vec.data(), _data, dim_s * sizeof(data_t),
                               cudaMemcpyDeviceToHost));
}

/** 
 * @brief Pushes the data from the provided std::array to the Array on the GPU
*/
template <util::Numeric data_t, size_t dim_s>
void Vec<data_t, dim_s, ArenaType::GPU>::push(
    std::array<data_t, dim_s> const& vec)
{
  assert(_data != nullptr);
  util::check_error(cudaMemcpy(_data, vec.data(), dim_s * sizeof(data_t),
                               cudaMemcpyHostToDevice));
}

/**
 * @brief Pulls the data from the gpu into the provided std::vector
*/
template <util::Numeric data_t, size_t dim_s>
void Vec<data_t, dim_s, ArenaType::GPU>::pull(std::vector<data_t>& vec) const
{
  assert(_data != nullptr);
  assert(vec.dim_s() == dim_s);
  util::check_error(cudaMemcpy(vec.data(), _data, dim_s * sizeof(data_t),
                               cudaMemcpyDeviceToHost));
}

/** 
 * @brief Pushes the data from the provided std::vector to the Array on the GPU
*/
template <util::Numeric data_t, size_t dim_s>
void Vec<data_t, dim_s, ArenaType::GPU>::push(std::vector<data_t> const& vec)
{
  assert(_data != nullptr);
  assert(vec.dim_s() == dim_s);
  util::check_error(cudaMemcpy(_data, vec.data(), dim_s * sizeof(data_t),
                               cudaMemcpyHostToDevice));
}

/**
 * @brief Piecewise adds two Arrays into a new Array
 * 
 * @tparam dim_s 
 * @param other 
 * @return Vec<data_t, dim_s, ArenaType::GPU> 
 */
template <util::Numeric data_t, size_t dim_s>
constexpr auto Vec<data_t, dim_s, ArenaType::GPU>::operator+(
    Vec<data_t, dim_s, ArenaType::GPU> const& other) const
    -> Vec<data_t, dim_s, ArenaType::GPU>
{
  Vec<data_t, dim_s, ArenaType::GPU> result{};
  vector_add_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_data, other.data(),
                                               result.data(), dim_s);
  util::check_error(cudaGetLastError());
  return result;
}

/**
 * @brief Piecewise adds second Array to first
 * 
 * @tparam dim_s 
 * @param other 
 * @return Vec<data_t, dim_s, ArenaType::GPU>& 
 */
template <util::Numeric data_t, size_t dim_s>
constexpr auto Vec<data_t, dim_s, ArenaType::GPU>::operator+=(
    Vec<data_t, dim_s, ArenaType::GPU> const& other)
    -> Vec<data_t, dim_s, ArenaType::GPU>&
{
  vector_add_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_data, other.data(), _data,
                                               dim_s);
  util::check_error(cudaGetLastError());
  return *this;
}

/**
 * @brief Piecewise subtracts two Arrays into a new Array
 * 
 * @tparam dim_s 
 * @param other 
 * @return Vec<data_t, dim_s, ArenaType::GPU> 
 */
template <util::Numeric data_t, size_t dim_s>
constexpr auto Vec<data_t, dim_s, ArenaType::GPU>::operator-(
    Vec<data_t, dim_s, ArenaType::GPU> const& other) const
    -> Vec<data_t, dim_s, ArenaType::GPU>
{
  Vec<data_t, dim_s, ArenaType::GPU> result{};
  vector_sub_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_data, other.data(),
                                               result.data(), dim_s);
  util::check_error(cudaGetLastError());
  return result;
}

/**
 * @brief Piecewise subtracts second Array from first
 * 
 * @tparam dim_s 
 * @param other 
 * @return Vec<data_t, dim_s, ArenaType::GPU>& 
 */
template <util::Numeric data_t, size_t dim_s>
constexpr auto Vec<data_t, dim_s, ArenaType::GPU>::operator-=(
    Vec<data_t, dim_s, ArenaType::GPU> const& other)
    -> Vec<data_t, dim_s, ArenaType::GPU>&
{
  vector_sub_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_data, other.data(), _data,
                                               dim_s);
  util::check_error(cudaGetLastError());
  return *this;
}

/**
 * @brief Hadamard product of two Arrays into a new Array
 * 
 * @tparam dim_s 
 * @param other 
 * @return Vec<data_t, dim_s, ArenaType::GPU> 
 */
template <util::Numeric data_t, size_t dim_s>
constexpr auto Vec<data_t, dim_s, ArenaType::GPU>::operator*(
    Vec<data_t, dim_s, ArenaType::GPU> const& other) const
    -> Vec<data_t, dim_s, ArenaType::GPU>
{
  Vec<data_t, dim_s, ArenaType::GPU> result{};
  vector_mul_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_data, other.data(),
                                               result.data(), dim_s);
  util::check_error(cudaGetLastError());

  return result;
}

/**
 * @brief Hadamard product of two Arrays into the first Array
 * 
 * @tparam dim_s 
 * @param other 
 * @return Vec<data_t, dim_s, ArenaType::GPU>& 
 */
template <util::Numeric data_t, size_t dim_s>
constexpr auto Vec<data_t, dim_s, ArenaType::GPU>::operator*=(
    Vec<data_t, dim_s, ArenaType::GPU> const& other)
    -> Vec<data_t, dim_s, ArenaType::GPU>&
{
  vector_mul_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_data, other.data(), _data,
                                               dim_s);
  util::check_error(cudaGetLastError());
  return *this;
}

/**
 * @brief Multiplies all values in the Array by a scalar into a new Array
 * 
 * @tparam dim_s 
 * @param scalar 
 * @return Vec<data_t, dim_s, ArenaType::GPU> 
 */
template <util::Numeric data_t, size_t dim_s>
constexpr auto Vec<data_t, dim_s, ArenaType::GPU>::operator*(
    data_t scalar) const -> Vec<data_t, dim_s, ArenaType::GPU>
{
  Vec<data_t, dim_s, ArenaType::GPU> result{};
  vector_scalar_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_data, scalar, result.data(),
                                                  dim_s);
  util::check_error(cudaGetLastError());
  return result;
}

/**
 * @brief Multiplies all values in the Array by a scalar
 * 
 * @tparam dim_s 
 * @param scalar 
 * @return Vec<data_t, dim_s, ArenaType::GPU>& 
 */
template <util::Numeric data_t, size_t dim_s>
constexpr auto Vec<data_t, dim_s, ArenaType::GPU>::operator*=(data_t scalar)
    -> Vec<data_t, dim_s, ArenaType::GPU>&
{
  vector_scalar_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(_data, scalar, _data, dim_s);
  util::check_error(cudaGetLastError());
  return *this;
}

// Constructors and Destructor

template <util::Numeric data_t, size_t dim_s>
Vec<data_t, dim_s, ArenaType::GPU>::Vec(
    Vec<data_t, dim_s, ArenaType::CPU>&& vec)
{
  if ( _data == nullptr ) allocate();
  util::check_error(cudaMemcpy(_data, vec.data(), dim_s * sizeof(data_t),
                               cudaMemcpyHostToDevice));
}

template <util::Numeric data_t, size_t dim_s>
Vec<data_t, dim_s, ArenaType::GPU>::Vec(
    Vec<data_t, dim_s, ArenaType::GPU>&& other) noexcept
{
  if ( _data != nullptr ) cudaFreeAsync(_data, nullptr);
  _data = other._data;
  other._data = nullptr;
}

template <util::Numeric data_t, size_t dim_s>
auto Vec<data_t, dim_s, ArenaType::GPU>::operator=(
    Vec<data_t, dim_s, ArenaType::GPU>&& other) noexcept
    -> Vec<data_t, dim_s, ArenaType::GPU>&
{
  if ( this == &other ) return *this;
  if ( _data != nullptr ) cudaFreeAsync(_data, nullptr);
  _data = other._data;
  other._data = nullptr;
  return *this;
}

template <util::Numeric data_t, size_t dim_s>
Vec<data_t, dim_s, ArenaType::GPU>::Vec(
    Vec<data_t, dim_s, ArenaType::GPU> const& other)
{
  if ( _data == nullptr ) allocate();
  util::check_error(cudaMemcpy(_data, other._data, dim_s * sizeof(data_t),
                               cudaMemcpyDeviceToDevice));
}

template <util::Numeric data_t, size_t dim_s>
auto Vec<data_t, dim_s, ArenaType::GPU>::operator=(
    Vec<data_t, dim_s, ArenaType::GPU> const& other)
    -> Vec<data_t, dim_s, ArenaType::GPU>&
{
  if ( this == &other ) return *this;
  if ( _data == nullptr ) allocate();
  util::check_error(cudaMemcpy(_data, other._data, dim_s * sizeof(data_t),
                               cudaMemcpyDeviceToDevice));
  return *this;
}

template <util::Numeric data_t, size_t dim_s>
Vec<data_t, dim_s, ArenaType::GPU>::Vec()
{
  allocate();
}

template <util::Numeric data_t, size_t dim_s>
Vec<data_t, dim_s, ArenaType::GPU>::~Vec()
{
  if ( _data == nullptr ) return;
  util::check_error(cudaFreeAsync(_data, nullptr));
}

template <util::Numeric data_t, size_t dim_s>
void Vec<data_t, dim_s, ArenaType::GPU>::push(
    Vec<data_t, dim_s, ArenaType::CPU> const& vec)
{
  assert(_data != nullptr);
  util::check_error(cudaMemcpy(_data, vec.data(), dim_s * sizeof(data_t),
                               cudaMemcpyHostToDevice));
}

template <util::Numeric data_t, size_t dim_s>
Vec<data_t, dim_s, ArenaType::GPU>::Vec(
    Vec<data_t, dim_s, ArenaType::CPU> const& vec)
{
  if ( _data == nullptr ) allocate();
  util::check_error(cudaMemcpy(_data, vec.data(), dim_s * sizeof(data_t),
                               cudaMemcpyHostToDevice));
}

}  // namespace cntns

#endif