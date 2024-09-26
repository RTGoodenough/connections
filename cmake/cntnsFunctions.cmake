
function(cntns_use_cuda)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  target_compile_definitions(
    ${PROJECT_NAME}
    PRIVATE
    CNTNS_USE_CUDA=1
  )
endfunction(cntns_use_cuda)