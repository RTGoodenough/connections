
#pragma once

namespace cntns {
__global__ void vector_randomize_kernel(float* data, float min, float max, int size);
__global__ void vector_fill_kernel(float* data, float value, int size);
__global__ void vector_reset_kernel(float* data, int size);

__global__ void max_element_kernel(float* input, int* maxIndex, int size);

__global__ void vector_add_kernel(float const* vecA, float const* vecB, float* out, int size);
__global__ void vector_sub_kernel(float const* vecA, float const* vecB, float* out, int size);
__global__ void vector_mul_kernel(float const* vecA, float const* vecB, float* out, int size);
__global__ void vector_scalar_kernel(float const* vecA, float vecB, float* out, int size);
__global__ void outer_product_kernel(float const* lhs, float const* rhs, float* result, int lhs_size, int rhs_size);
}