

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cufftXt.h>

namespace cntns {

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)

/**
 * @brief Gets the index of the element in the matrix
 * 
 * @param row 
 * @param col 
 * @param cols 
 * @return unsigned int 
 */
__device__ auto index(unsigned int row, unsigned int col, int cols) -> unsigned int {
  return row * cols + col;
}

/**
 * @brief Fill the matrix with random values between min and max
 * 
 * @param data 
 * @param min 
 * @param max 
 * @param rows 
 * @param cols 
 * @return __global__ 
 */
__global__ void mat_randomize_kernel(float* data, float min, float max, int rows, int cols) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  curandState state;
  curand_init(static_cast<uint64_t>(clock()) + row + col, 0, 0, &state);

  if (row < rows && col < cols) {
    data[index(row, col, cols)] = curand_uniform(&state) * (max - min) + min;
  }
}

/**
 * @brief Fill the matrix with the given value
 * 
 * @param data 
 * @param min 
 * @param max 
 * @param rows 
 * @param cols 
 * @return __global__ 
 */
__global__ void mat_fill_kernel(float* data, float value, int rows, int cols) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    data[index(row, col, cols)] = value;
  }
}

/**
 * @brief Adds two matrices together
 * 
 * @param A 
 * @param B 
 * @param out 
 * @param rows 
 * @param cols 
 * @return __global__ 
 */
__global__ void mat_add_kernel(float const* A, float const* B, float* out, int rows, int cols) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    out[index(row, col, cols)] = A[index(row, col, cols)] + B[index(row, col, cols)];
  }
}

/**
 * @brief Transposes the matrix
 * 
 * @param input 
 * @param output 
 * @param inRows 
 * @param inCols 
 * @return __global__ 
 */
__global__ void transpose_kernel(float const* input, float* output, int inRows, int inCols) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < inRows && col < inCols) {
    // NOLINTNEXTLINE(readability-suspicious-call-argument) this is correct
    output[index(col, row, inRows)] = input[index(row, col, inCols)];
  }
}

/**
 * @brief Adds the outer product of two vectors to a matrix
 * 
 * @param A 
 * @param B 
 * @param out 
 * @param rows 
 * @param cols 
 * @return __global__ 
 */
__global__ void mat_outer_product_add_kernel(float * A, float const* lhs, float const* rhs, int rows, int cols) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    A[index(row, col, cols)] += lhs[row] * rhs[col];
  }
}

/**
 * @brief Subtracts two matrices
 * 
 * @param A 
 * @param B 
 * @param out 
 * @param rows 
 * @param cols 
 * @return __global__ 
 */
__global__ void mat_sub_kernel(float const* A, float const* B, float* out, int rows, int cols) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    out[index(row, col, cols)] = A[index(row, col, cols)] - B[index(row, col, cols)];
  }
}

/**
 * @brief Piecewise Multiplies two matrices
 * 
 * @param A 
 * @param B 
 * @param out 
 * @param rows 
 * @param cols 
 * @return __global__ 
 */
__global__ void mat_mul_kernel(float const* A, float const* B, float* out, int rows, int cols) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    out[index(row, col, cols)] = A[index(row, col, cols)] * B[index(row, col, cols)];
  }
}

/**
 * @brief Multiplies a matrix by a scalar
 * 
 * @param A 
 * @param B 
 * @param out 
 * @param rows 
 * @param cols 
 * @param colsB 
 * @return __global__ 
 */
__global__ void mat_scalar_kernel(float const* A, float B, float* out, int rows, int cols) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    out[index(row, col, cols)] = A[index(row, col, cols)] * B;
  }
}

/**
 * @brief Multiplies a matrix by a scalar in place
 * 
 * @param A 
 * @param B 
 * @param rows 
 * @param cols 
 * @return __global__ 
 */
__global__ void mat_scalar_eq_kernel(float* A, float B, int rows, int cols) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    A[index(row, col, cols)] = A[index(row, col, cols)] * B;
  }
}

/**
 * @brief Multiplies a matrix by a vector
 * 
 * @param A 
 * @param B 
 * @param out 
 * @param rows 
 * @param cols 
 * @return __global__ 
 */
__global__ void mat_vec_mul_kernel(float const* A, float const* B, float* out, int rows, int cols) {
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < rows) {
    float sum = 0.0F;
    for (int i = 0; i < cols; ++i) {
      sum +=  A[index(row, i, cols)] * B[i];
    }
    out[row] = sum;
  }
}

/**
 * @brief Multiplies the transpose of a matrix by a vector
 * 
 * @param A 
 * @param B 
 * @param out 
 * @param rows 
 * @param cols 
 * @return __global__ 
 */
__global__ void mat_transpose_vec_mul_kernel(float const* A, float const* B, float* out, int rows, int cols) {
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < cols) {
    float sum = 0.0F;
    for (int i = 0; i < rows; ++i) {
      sum +=  A[index(i, row, cols)] * B[i];
    }
    out[row] = sum;
  }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)

}  // namespace cntns