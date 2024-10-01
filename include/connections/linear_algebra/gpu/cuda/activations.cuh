#pragma once

namespace cntns {
__global__ void logsig_kernel(double const* input, double* output, int size);
__global__ void logsig_derivative_kernel(double const* input, double* output, int size);

__global__ void relu_kernel(double const* input, double* output, int size);
__global__ void relu_derivative_kernel(double const* input, double* output, int size);

__global__ void softmax_kernel(double const* input, double* output, int size);
}

