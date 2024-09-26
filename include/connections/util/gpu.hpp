#pragma once

#ifdef CNTNS_USE_CUDA

#include <cstddef>
#include <cstdio>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

namespace cntns::util {

/**
 * @brief Checks the status of a CUDA operation
 * 
 */
inline void check_error(cudaError_t status)
{
  if ( status != cudaSuccess ) {
    std::printf("CUDA Error: %s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

/**
 * @brief Gets the percentage of GPU memory used
 * 
 * @return auto 
 */
inline auto memory_usage()
{
  size_t freeByte = 0;
  size_t totalByte = 0;

  auto cudaStatus = cudaMemGetInfo(&freeByte, &totalByte);

  if ( cudaSuccess != cudaStatus ) {
    throw std::runtime_error("Failed to get GPU memory info");
  }

  float used = static_cast<float>(totalByte - freeByte) /
               static_cast<float>(totalByte) * 100.0F;

  return used;
}
}  // namespace cntns::util

#endif