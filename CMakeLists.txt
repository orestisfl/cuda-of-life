project(cuda_of_life)
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

FILE(GLOB SOURCES "*.cu" "*.cpp" "*.h")
set(CUDA_64_BIT_DEVICE_CODE ON)

if (${CUDA_VERSION_MAJOR} GREATER 5)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DCUDA_65)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20 --compiler-options -Wall)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES})
add_executable(gen.out generate_table.cxx utils.cpp utils.h)
