#pragma once

namespace cntns {

__global__ void mat_randomize_kernel(double* data, double min, double max, int rows, int cols);
__global__ void mat_fill_kernel(double* data, double value, int rows, int cols);
__global__ void transpose_kernel(double const* input, double* output, int rows, int cols);

__global__ void mat_add_kernel(double const* A, double const* B, double* out, int rows, int cols);
__global__ void mat_outer_product_add_kernel(double * A, double const* lhs, double const* rhs, int rows, int cols);

__global__ void mat_sub_kernel(double const* A, double const* B, double* out, int rows, int cols);

__global__ void mat_mul_kernel(double const* A, double const* B, double* out, int rows, int cols);

__global__ void mat_scalar_kernel(double const* A, double B, double* out, int rows, int cols);

__global__ void mat_vec_mul_kernel(double const* A, double const* B, double* out, int rows, int cols);

__global__ void mat_transpose_vec_mul_kernel(double const* A, double const* B, double* out, int rows, int cols);


}  // namespace cntns