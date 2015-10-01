project(cuda_of_life)
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "building for ${CMAKE_BUILD_TYPE}")

FILE(GLOB SOURCES "*.cu" "*.cpp" "*.h")
set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_VERBOSE_BUILD ON)

LIST(APPEND CMAKE_CXX_FLAGS "-ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)
LIST(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)

if (${CUDA_VERSION_MAJOR} GREATER 5)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DCUDA_65)
endif()

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES})
add_executable(gen.out generate_table.cxx utils.cpp utils.h)
