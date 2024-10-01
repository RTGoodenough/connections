
#include <cfloat>
#include <limits>

namespace cntns {

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)

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

__global__ void relu_kernel(double const* input, double* output, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    output[idx] = input[idx] >= 0 ? input[idx] : 0.0;
  }
}

__global__ void relu_derivative_kernel(double const* input, double* output, int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    output[idx] = input[idx] >= 0 ? 1.0 : 0.0;;
  }
}

__global__ void softmax_kernel(double const* input, double* output, int size) {
    extern __shared__ double sharedData[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Find the maximum value in the input for numerical stability
    double maxVal = -FLT_MAX;
    if (index < size) {
        maxVal = input[index];
    }

    // Store maximum value in shared memory
    sharedData[tid] = maxVal;
    __syncthreads();

    // Perform parallel reduction to find the maximum value
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }
    maxVal = sharedData[0];

    // Step 2: Compute the exponentials and their sum
    double sumExp = 0.0;
    if (index < size) {
        sharedData[tid] = exp(input[index] - maxVal); // Subtract max_val for numerical stability
        sumExp = sharedData[tid];
    }
    __syncthreads();

    // Perform parallel reduction to find the sum of exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    sumExp = sharedData[0];

    // Step 3: Compute Softmax by dividing each exponential by the sum of exponentials
    if (index < size) {
        output[index] = sharedData[tid] / sumExp;
    }
}


// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)
}  // namespace cntns