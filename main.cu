#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

static inline void cudaCheckErrors(const char msg[], const char file[], int line) {
    do {
        cudaError_t __err = cudaGetLastError();

        if (__err != cudaSuccess) {
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",
                    msg, cudaGetErrorString(__err),
                    file, line);
            cudaDeviceReset();
            exit(1);
        }
    } while (0);
}

#define POS(i,j) (i + WIDTH * j)

#define SET_BIT(val, bit_idx) val |= (ONE << bit_idx)
#define SET_BOARD(val, i, j) SET_BIT(val, POS(i, j))
#define CLEAR_BIT(val, bit_idx) val &= ~(ONE << bit_idx)
#define CLEAR_BOARD(val, i, j) CLEAR_BIT(val, POS(i, j))
#define TOGGLE_BIT(val, bit_idx) val ^= (ONE << bit_idx)
#define TOGGLE_BOARD(val, i, j) TOGGLE_BIT(val, POS(i, j))
#define BIT_IS_SET(val, bit_idx) (val & (ONE << bit_idx))
#define BOARD_IS_SET(val, i, j) BIT_IS_SET(val, POS(i, j))

__global__
void convert_to_tiled(int* d_table, bboard* d_board, size_t dim, size_t dim_board) {
    int major_i = blockIdx.y * blockDim.y + threadIdx.y;
    int major_j = blockIdx.x * blockDim.x + threadIdx.x;
    int board_i = 0;
    int board_j = 0;
    int real_i = major_i * WIDTH + board_i;
    int real_j = major_j * WIDTH + board_j;
//    int idx = row * WIDTH + col;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("usage: %s fname dim (iter blockx blocky gridx gridy)\n", argv[0]);
        exit(1);
    }
    int n_runs = DFL_RUNS;
    if (argc >= 4) {
        n_runs = atoi(argv[3]);
    }
    const size_t dim = atoi(argv[2]);
    const size_t total_elements = dim * dim;
    const size_t mem_size = total_elements * sizeof(int);
    dim3 block;
    dim3 grid;

    if (argc >= 6) {
        block.x = atoi(argv[4]);
        block.y = block.x;
        grid.x = atoi(argv[6]);
        grid.y = grid.x;
    }

    char* filename = argv[1];
    int* table;
    table = (int*) malloc(mem_size);
    read_from_file(table, filename, dim);

    bboard* d_board;
    int* d_table;
    const size_t dim_board = CEIL_DIV(dim, WIDTH);
    const size_t mem_size_board = dim_board * dim_board * sizeof(bboard);
    cudaMalloc((void**) &d_table,  mem_size);
    cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);
    cudaMalloc((void**) &d_board, mem_size_board);
    cudaCheckErrors("device allocation of GOL tiled matrix failed", __FILE__, __LINE__);

    bboard* devPtr;
    size_t pitch;
    cudaMallocPitch((void**)&devPtr, &pitch, dim_board * sizeof(bboard), dim_board);

    cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);

    convert_to_tiled <<< grid, block >>> (d_table, d_board, dim, dim_board);

    return 0;
}

