#pragma once

namespace cntns {

__global__ void mat_randomize_kernel(float* data, float min, float max, int rows, int cols);
__global__ void mat_fill_kernel(float* data, float value, int rows, int cols);
__global__ void transpose_kernel(float const* input, float* output, int rows, int cols);

__global__ void mat_add_kernel(float const* A, float const* B, float* out, int rows, int cols);
__global__ void mat_outer_product_add_kernel(float * A, float const* lhs, float const* rhs, int rows, int cols);

__global__ void mat_sub_kernel(float const* A, float const* B, float* out, int rows, int cols);

__global__ void mat_mul_kernel(float const* A, float const* B, float* out, int rows, int cols);

__global__ void mat_scalar_kernel(float const* A, float B, float* out, int rows, int cols);

__global__ void mat_vec_mul_kernel(float const* A, float const* B, float* out, int rows, int cols);

__global__ void mat_transpose_vec_mul_kernel(float const* A, float const* B, float* out, int rows, int cols);


}  // namespace cntns