

namespace cntns {

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)


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

__global__ void cross_entropy(double const* correct, double const* result, double* loss, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread computes the loss for a single sample
  if (tid < size) {
    int label = static_cast<int>(correct[tid]);
    double predictedProb = result[tid * size + label];
    
    // Compute cross-entropy loss for this sample (assuming log base e)
    double sampleLoss = -log(predictedProb);

    // Accumulate the loss
    atomicAdd(loss, sampleLoss / static_cast<double>(size));
  }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-identifier-length)
}  // namespace cntns
