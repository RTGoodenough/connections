

namespace cntns {

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)


/**
 * @brief Calculates the quadratic cost function between correct and result
 * 
 */
__global__ void quadratic_cost_kernel(float const* correct, float const* result, float* loss, int size) {
  __shared__ float scratch[256];
  
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    float diff = correct[idx] - result[idx];
    scratch[idx] = diff * diff;
  }

  __syncthreads();

  if (idx == 0) {
    float sum = 0.0F;
    for (int i = 0; i < size; i++) {
      sum += scratch[i];
    }
    *loss = sum * (0.5 * static_cast<float>(size));
  }
}

__global__ void cross_entropy(float const* correct, float const* result, float* loss, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread computes the loss for a single sample
  if (tid < size) {
    int label = static_cast<int>(correct[tid]);
    float predictedProb = result[tid * size + label];
    
    // Compute cross-entropy loss for this sample (assuming log base e)
    float sampleLoss = -log(predictedProb);

    // Accumulate the loss
    atomicAdd(loss, sampleLoss / static_cast<float>(size));
  }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)
}  // namespace cntns
