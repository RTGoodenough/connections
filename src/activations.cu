
#include <cfloat>
#include <cinttypes>
#include <limits>

namespace cntns {

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)

/**
 * @brief Applies the logistic sigmoid function to an array
 * 
 */
__global__ void logsig_kernel(float const* input, float* output, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    output[idx] = 1.0F / (1.0F + exp(-input[idx]));
  }
}

/**
 * @brief Applies the logistic sigmoid derivative to an array
 * 
 */
__global__ void logsig_derivative_kernel(float const* input, float* output, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    output[idx] = input[idx] * (1 - input[idx]);
  }
} 

__global__ void relu_kernel(float const* input, float* output, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    output[idx] = input[idx] >= 0 ? input[idx] : 0.0F;
  }
}

__global__ void relu_derivative_kernel(float const* input, float* output, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    output[idx] = input[idx] >= 0 ? 1.0 : 0.0;;
  }
}

__global__ void softmax_kernel(float const* input, float* output, int size) {
  // TODO(rolland): fix this
  unsigned int tid = threadIdx.x;

  if (tid == 0) {
    float max = -1;
    for (size_t i = 0; i < size; ++i) {
      if (input[i] > max) max = input[i];
    }

    float sum = 0;
    for (size_t i = 0; i < size; ++i) {
      output[i] =  exp(input[i] - max);
      sum += output[i];
    }

    for (size_t i = 0; i < size; ++i) {
      output[i] =  output[i] / sum;
    }
  }

  // __shared__ float sharedMemory[256];
  // unsigned int tid = threadIdx.x;
  // unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // sharedMemory[tid] = input[i];
  // __syncthreads();

  // // find the max value
  // for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
  //   if (tid < stride) {
  //     if (sharedMemory[tid + stride] > sharedMemory[tid]) {
  //       sharedMemory[tid] = sharedMemory[tid + stride];
  //     }
  //   }
  //   __syncthreads();
  // }
  // float maxVal = sharedMemory[0];

  // // set shared memory to 
  // if ( tid < size ) {
  //     sharedMemory[tid] = exp(input[tid] - maxVal);
  // }
  // __syncthreads();

  // // sum up values
  
  // for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
  //   if (tid < stride) {
  //     sharedMemory[tid] += sharedMemory[tid + stride];
  //   }
  //   __syncthreads();
  // }
  // float sumExp = sharedMemory[0];

  // if ( tid < size ) {
  //     output[tid] = sharedMemory[tid] / sumExp;
  // }
}


// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)
}  // namespace cntns