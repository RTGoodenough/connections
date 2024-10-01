#pragma once

namespace cntns {
__global__ void quadratic_cost_kernel(double const* correct, double const* result, double* loss, int size);

__global__ void cross_entropy(double const* correct, double const* result, double* loss, int size);
}