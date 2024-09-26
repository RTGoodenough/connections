
#pragma once

namespace cntns {
__global__ void vector_randomize_kernel(double* data, double min, double max, int size);
__global__ void vector_fill_kernel(double* data, double value, int size);
__global__ void vector_reset_kernel(double* data, int size);

__global__ void vector_add_kernel(double const* vecA, double const* vecB, double* out, int size);
__global__ void vector_sub_kernel(double const* vecA, double const* vecB, double* out, int size);
__global__ void vector_mul_kernel(double const* vecA, double const* vecB, double* out, int size);
__global__ void vector_scalar_kernel(double const* vecA, double vecB, double* out, int size);

__global__ void logsig_kernel(double const* input, double* output, int size);
__global__ void logsig_derivative_kernel(double const* input, double* output, int size);

__global__ void outer_product_kernel(double const* lhs, double const* rhs, double* result, int lhs_size, int rhs_size);
__global__ void quadratic_cost_kernel(double const* correct, double const* result, double* loss, int size);
}