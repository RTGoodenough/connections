
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

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


__global__ void max_element_kernel(double* input, int* maxIndex, int size) {
    extern __shared__ int sharedMemory[];

    // Thread index
    int tid = threadIdx.x;

    // Load elements into shared memory
    if (tid < size) {
        sharedMemory[tid] = input[tid];
    }
    __syncthreads();

    // Parallel reduction to find the index of maximum element
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0 && tid + stride < size) {
            if (sharedMemory[tid + stride] > sharedMemory[tid]) {
                sharedMemory[tid] = sharedMemory[tid + stride];
            }
        }
        __syncthreads();
    }

    // Store the result back to the maxIndex variable
    if (tid == 0) {
        *maxIndex = tid; // Assuming max element is at sharedMemory[tid]
    }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)
}  // namespace cntns