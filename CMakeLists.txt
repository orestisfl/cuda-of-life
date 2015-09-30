project(cuda_of_life)
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)
FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h")

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES})

