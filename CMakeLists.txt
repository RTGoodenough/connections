cmake_minimum_required(VERSION 3.14)

set(CNTNS_NAME cntns)

project(${CNTNS_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(
  ${CNTNS_NAME}
  STATIC
  src/matrix.cu
  src/vector.cu
  src/activations.cu
  src/loss.cu
)

target_include_directories(${CNTNS_NAME} INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${CNTNS_NAME}ConfigVersion.cmake"
    VERSION 0.1
    COMPATIBILITY AnyNewerVersion
)

install(TARGETS ${CNTNS_NAME}
    EXPORT ${CNTNS_NAME}Targets
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
    RUNTIME DESTINATION bin COMPONENT Runtime
    PUBLIC_HEADER DESTINATION include COMPONENT Development
    BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${CNTNS_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${CNTNS_NAME}Config.cmake"
    INSTALL_DESTINATION lib/cmake/${CNTNS_NAME}
)

install(EXPORT ${CNTNS_NAME}Targets DESTINATION lib/cmake/${CNTNS_NAME})
install(FILES "${PROJECT_BINARY_DIR}/${CNTNS_NAME}ConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/${CNTNS_NAME}Config.cmake"
              "cmake/${CNTNS_NAME}Functions.cmake"
        DESTINATION lib/cmake/${CNTNS_NAME})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
