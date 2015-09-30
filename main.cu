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

__global__ void calculate_next_generation(void){}

#define DEFAULT_OPTY 16
#define DEFAULT_OPTX 16
void best_block_size(int* optx, int* opty) {
    #ifdef CUDA_65
    // The launch configurator returned block size
    int block_size = 0;
    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int min_grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       (void*)calculate_next_generation,  0, 0);

    *optx = (int) ceil(sqrt(block_size));

    while (block_size % *optx) {
        (*optx)--;
    }

    *opty = block_size / *optx;
    #else
    *optx = DEFAULT_OPTX;
    *opty = DEFAULT_OPTY;
    #endif
}

// fill array A with zeros
__global__
void zero_k(int* A, size_t dim) {
    for (int i = 0; i < dim * dim; i ++) {
        A[i] = 0;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
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
    const size_t dim_board = CEIL_DIV(dim, WIDTH);
    const size_t mem_size_board = dim_board * dim_board * sizeof(bboard);

    if (argc >= 6) {
        block.x = atoi(argv[4]);
        block.y = block.x;
        grid.x = atoi(argv[5]);
        grid.y = grid.x;
    } else {
        int optx, opty;
        best_block_size(&optx, &opty);
        fprintf(stderr, "opt=%d %d\n", optx, opty);

        block.x = (dim_board < (uint)optx) ? dim_board : optx;
        block.y = (dim_board < (uint)opty) ? dim_board : opty;
        grid.x = CEIL_DIV(dim_board, block.x);
        grid.y = CEIL_DIV(dim_board, block.y);
    }

    char* filename = argv[1];
    fprintf(stderr,
            "%s: Reading %zux%zu table from file %s\n", argv[0], dim, dim, filename);
    fprintf(stderr,
            "%s: Running on a grid(%d, %d) with a block(%d, %d):\nFilename: %s with dim %zu for %d iterations\n",
            argv[0], grid.x, grid.y, block.x, block.y, filename, dim, n_runs);
    int* table;
    table = (int*) malloc(mem_size);
    //    print_table(table, dim);
    read_from_file(table, filename, dim);
    //    print_table(table, dim);

    int* d_table;
    cudaMalloc((void**) &d_table,  mem_size);
    cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);

    bboard* d_board;
    size_t pitch;
    cudaMallocPitch((void**)&d_board, &pitch, dim_board * sizeof(bboard), dim_board);
    cudaCheckErrors("device pitch allocation of GOL matrix failed", __FILE__, __LINE__);

    cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);
    free(table);

    convert_to_tiled <<< grid, block >>> (d_table, d_board, dim, dim_board, pitch);
    cudaCheckErrors("convert_to_tiled failed", __FILE__, __LINE__);

//    zero_k <<< 1, 1 >>> (d_table, dim);

    convert_from_tiled <<< grid, block >>> (d_table, d_board, dim, dim_board, pitch);
    cudaCheckErrors("convert_from_tiled failed", __FILE__, __LINE__);

    table = (int*) malloc(mem_size);
    cudaMemcpy(table, d_table, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy from device to host memory failed", __FILE__, __LINE__);

    //    print_table(table, dim);
    save_table(table, dim, "test_results.bin");

    UNUSED(mem_size);
    UNUSED(mem_size_board);

    return 0;
}
