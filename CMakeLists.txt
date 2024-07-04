cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME connections)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)

project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(
  ${PROJECT_NAME} 
  STATIC
  ${SRC_DIR}/connections.cpp
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  ${INCLUDE_DIR}
)

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
  ${DEPS_DIR}
)

find_package(Eigen3 REQUIRED NO_MODULE)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
  Eigen3::Eigen
)