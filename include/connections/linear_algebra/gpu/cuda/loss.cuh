#pragma once

namespace cntns {
__global__ void quadratic_cost_kernel(float const* correct, float const* result, float* loss, int size);

__global__ void cross_entropy(float const* correct, float const* result, float* loss, int size);
}