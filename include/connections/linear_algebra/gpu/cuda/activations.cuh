#pragma once

namespace cntns {
__global__ void logsig_kernel(float const* input, float* output, int size);
__global__ void logsig_derivative_kernel(float const* input, float* output, int size);

__global__ void relu_kernel(float const* input, float* output, int size);
__global__ void relu_derivative_kernel(float const* input, float* output, int size);

__global__ void softmax_kernel(float const* input, float* output, int size);
}

