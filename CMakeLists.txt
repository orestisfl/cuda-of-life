project(cuda_of_life)
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

execute_process(COMMAND nvcc -x cu ${CMAKE_CURRENT_SOURCE_DIR}/devInfo.c -o ${CMAKE_BINARY_DIR}/devInfo.out)
execute_process(COMMAND ./devInfo.out OUTPUT_VARIABLE CUDA_ARCH)
execute_process(COMMAND rm -f ${CMAKE_BINARY_DIR}/devInfo.out)
message("Cuda architecture:  ${CUDA_ARCH}")
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "building for ${CMAKE_BUILD_TYPE}")

FILE(GLOB SOURCES "*.cu" "*.cpp" "*.h")
set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_VERBOSE_BUILD ON)

LIST(APPEND CMAKE_CXX_FLAGS "-ffast-math -Wall -Wextra")
LIST(APPEND CMAKE_CXX_FLAGS_RELEASE "-ffast-math -Wall -Wextra -O3")
LIST(APPEND CMAKE_C_FLAGS_RELEASE "-ffast-math -Wall -Wextra -O3")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -use_fast_math)
LIST(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)
LIST(APPEND CUDA_NVCC_FLAGS_DEBUG -O0 -g -G -pg -lineinfo -Xptxas -v)

if (${CUDA_VERSION_MAJOR} GREATER 5)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DCUDA_65)
endif()

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES})
add_executable(gen.out generate_table.cxx utils.cpp utils.h)
