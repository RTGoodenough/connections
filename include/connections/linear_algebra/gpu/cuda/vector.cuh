
#pragma once

namespace cntns {
__global__ void vector_randomize_kernel(double* data, double min, double max, int size);
__global__ void vector_fill_kernel(double* data, double value, int size);
__global__ void vector_reset_kernel(double* data, int size);

__global__ void max_element_kernel(double* input, int* maxIndex, int size);

__global__ void vector_add_kernel(double const* vecA, double const* vecB, double* out, int size);
__global__ void vector_sub_kernel(double const* vecA, double const* vecB, double* out, int size);
__global__ void vector_mul_kernel(double const* vecA, double const* vecB, double* out, int size);
__global__ void vector_scalar_kernel(double const* vecA, double vecB, double* out, int size);
__global__ void outer_product_kernel(double const* lhs, double const* rhs, double* result, int lhs_size, int rhs_size);
}