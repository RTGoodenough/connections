
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <limits>
#include <cfloat>
#include <curand.h>
#include <curand_kernel.h>

namespace cntns {

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)

/**
 * @brief Fills an array with random values between min and max.
 * 
 */
__global__ void vector_randomize_kernel(float* data, float min, float max, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandState state;
  curand_init(static_cast<uint64_t>(clock()), idx, 0, &state);

  if (idx < size) {
    data[idx] = curand_uniform(&state) * (max - min) + min;
  }
}

/**
 * @brief Fills an array with a given value
 * 
 */
__global__ void vector_fill_kernel(float* data, float value, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    data[idx] = value;
  }
}

/**
 * @brief Resets an array to 0
 * 
 */
__global__ void vector_reset_kernel(float* data, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    data[idx] = 0.0;
  }
}

/**
 * @brief Adds two arrays together
 * 
 */
__global__ void vector_add_kernel(float const* A, float const* B, float* out, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = A[idx] + B[idx];
  }
}

/**
 * @brief Subtracts two arrays
 * 
 */
__global__ void vector_sub_kernel(float const* A, float const* B, float* out, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = A[idx] - B[idx];
  }
}

/**
 * @brief Piecewise Multiplies two arrays
 * 
 */
__global__ void vector_mul_kernel(float const* A, float const* B, float* out, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = A[idx] * B[idx];
  }
}

/**
 * @brief Multiplies an array by a scalar
 * 
 */
__global__ void vector_scalar_kernel(float const*A, float B, float* out, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = A[idx] * B;
  }
}

/**
 * @brief Computes the outer product of two arrays
 * 
 */
__global__ void outer_product_kernel(float const* lhs, float const* rhs, float* result, unsigned int dim,
                                     unsigned int other_dim) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < dim && col < other_dim) {
    result[col * dim + row] = lhs[row] * rhs[col];
  }
}


__global__ void max_element_idx_kernel(float const* input, size_t* maxIndex, int size) {
    __shared__ unsigned int sharedMemory[256];
    unsigned int tid = threadIdx.x;

    if (tid < size) {
        sharedMemory[tid] = tid;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        if (input[sharedMemory[tid] + stride] > input[sharedMemory[tid]]) {
          sharedMemory[tid] = tid + stride;
        }  
    }
    __syncthreads();
  }

  if (tid == 0) {
      *maxIndex = sharedMemory[0];
  }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)
}  // namespace cntns