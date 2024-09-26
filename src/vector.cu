

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>

namespace cntns {

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)

/**
 * @brief Fills an array with random values between min and max.
 * 
 */
__global__ void vector_randomize_kernel(double* data, double min, double max, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandState state;
  curand_init(static_cast<uint64_t>(clock()) + idx, 0, 0, &state);

  if (idx < size) {
    data[idx] = curand_uniform(&state) * (max - min) + min;
  }
}

/**
 * @brief Fills an array with a given value
 * 
 */
__global__ void vector_fill_kernel(double* data, double value, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    data[idx] = value;
  }
}

/**
 * @brief Resets an array to 0
 * 
 */
__global__ void vector_reset_kernel(double* data, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    data[idx] = 0.0;
  }
}

/**
 * @brief Adds two arrays together
 * 
 */
__global__ void vector_add_kernel(double const* A, double const* B, double* out, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = A[idx] + B[idx];
  }
}

/**
 * @brief Subtracts two arrays
 * 
 */
__global__ void vector_sub_kernel(double const* A, double const* B, double* out, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = A[idx] - B[idx];
  }
}

/**
 * @brief Piecewise Multiplies two arrays
 * 
 */
__global__ void vector_mul_kernel(double const* A, double const* B, double* out, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = A[idx] * B[idx];
  }
}

/**
 * @brief Multiplies an array by a scalar
 * 
 */
__global__ void vector_scalar_kernel(double const*A, double B, double* out, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = A[idx] * B;
  }
}

/**
 * @brief Computes the outer product of two arrays
 * 
 */
__global__ void outer_product_kernel(double const* lhs, double const* rhs, double* result, unsigned int dim,
                                     unsigned int other_dim) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < dim && col < other_dim) {
    result[col * dim + row] = lhs[row] * rhs[col];
  }
}

/**
 * @brief Applies the logistic sigmoid function to an array
 * 
 */
__global__ void logsig_kernel(double const* input, double* output, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    output[idx] = 1.0F / (1.0F + exp(-input[idx]));
  }
}

/**
 * @brief Applies the logistic sigmoid derivative to an array
 * 
 */
__global__ void logsig_derivative_kernel(double const* input, double* output, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    output[idx] = input[idx] * (1 - input[idx]);
  }
}

/**
 * @brief Calculates the quadratic cost function between correct and result
 * 
 */
__global__ void quadratic_cost_kernel(double const* correct, double const* result, double* loss, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double scratch[256];


  if (idx < size) {
    double diff = correct[idx] - result[idx];
    scratch[idx] = diff * diff;
  }

  __syncthreads();

  if (idx == 0) {
    double sum = 0.0F;
    for (int i = 0; i < size; i++) {
      sum += scratch[i];
    }
    *loss = sum * (0.5 * static_cast<double>(size));
  }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)
}  // namespace cntns