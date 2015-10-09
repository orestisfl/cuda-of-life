#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

static void swap_boards(bboard** a, bboard** b) {
    bboard* t;
    t = *a;
    *a = *b;
    *b = t;
}

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
    *opty = *optx;
    #else
    *optx = DEFAULT_OPTX;
    *opty = DEFAULT_OPTY;
    #endif
}

#ifndef TESTING
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s fname dim (iter blockx blocky gridx gridy)\n", argv[0]);
        exit(1);
    }
    int n_runs = DFL_RUNS;
    if (argc >= 4) {
        n_runs = atoi(argv[3]);
    }
    const int dim = atoi(argv[2]);
    const size_t total_elements = dim * dim;
    const size_t mem_size = total_elements * sizeof(int);
    dim3 block;
    dim3 grid;
    const int dim_board_w = CEIL_DIV(dim, WIDTH);
    const int dim_board_h = CEIL_DIV(dim, HEIGHT);

    if (argc >= 7) {
        block.x = atoi(argv[4]);
        block.y = atoi(argv[5]);
        grid.x = atoi(argv[6]);
        grid.y = atoi(argv[7]);
    } else {
        int optx, opty;
        best_block_size(&optx, &opty);
        fprintf(stderr, "opt=%d %d\n", optx, opty);

        block.x = (dim_board_h < optx) ? dim_board_h : optx;
        block.y = (dim_board_w < opty) ? dim_board_w : opty;
        grid.x = CEIL_DIV(dim_board_h, block.x);
        grid.y = CEIL_DIV(dim_board_w, block.y);
    }
    cudaFree(0); // init device

    //    const int remaining_blocks = remaining_dim / WIDTH; // for shared??
    const int remaining_dim_h = grid.x * block.x * HEIGHT - dim;
    const int remaining_dim_w = grid.y * block.y * WIDTH - dim;
    const int remaining_cells_h = remaining_dim_h % HEIGHT;
    const int remaining_cells_w = remaining_dim_w % WIDTH;

    char* filename = argv[1];
    fprintf(stderr,
            "%s: Reading %dx%d table from file %s\n", argv[0], dim, dim, filename);
    fprintf(stderr,
            "%s: Running on a grid(%d, %d) with a block(%d, %d):\nFilename: %s with dim %d for %d iterations\n",
            argv[0], grid.x, grid.y, block.x, block.y, filename, dim, n_runs);
    int* table;
    table = (int*) malloc(mem_size);
    read_from_file(table, filename, dim);
    //    print_table(table, dim);

    int* d_table;
    cudaMalloc((void**) &d_table,  mem_size);
    cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);

    bboard* d_board;
    bboard* d_help;
    size_t pitch;
    cudaMallocPitch((void**)&d_board, &pitch, dim_board_w * sizeof(bboard), dim_board_h);
    cudaCheckErrors("device pitch allocation of GOL matrix failed", __FILE__, __LINE__);
    cudaMallocPitch((void**)&d_help, &pitch, dim_board_w * sizeof(bboard), dim_board_h);
    cudaCheckErrors("device pitch allocation of GOL matrix failed", __FILE__, __LINE__);

    cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);
    free(table);

    convert_to_tiled <<< grid, block >>> (d_table, d_board, dim, pitch);
    cudaCheckErrors("convert_to_tiled failed", __FILE__, __LINE__);

    const bool no_rem = (remaining_cells_w == 0 && remaining_cells_h == 0);

    // start timewatch
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //    zero_k <<< 1, 1 >>> (d_table, dim);
    for (int i = 0; i < n_runs; ++i) {
        if (no_rem) {
            calculate_next_generation_no_rem <<< grid, block>>> (d_board, d_help,
                                                          dim, dim_board_w, dim_board_h, pitch);
        }
        else {
            calculate_next_generation <<< grid, block>>> (d_board, d_help,
                                                          dim, dim_board_w, dim_board_h, pitch,
                                                          remaining_cells_w, remaining_cells_h);
        }
        cudaCheckErrors("calculating next generation failed", __FILE__, __LINE__);
        swap_boards(&d_board, &d_help);
    }

    cudaStreamSynchronize(0);
    // end timewatch
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("%f\n", time);

    convert_from_tiled <<< grid, block >>> (d_table, d_board, dim, pitch);
    cudaCheckErrors("convert_from_tiled failed", __FILE__, __LINE__);

    table = (int*) malloc(mem_size);
    cudaMemcpy(table, d_table, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy from device to host memory failed", __FILE__, __LINE__);

    //    print_table(table, dim);
    save_table(table, dim, "test_results.bin");

    // reset gpu
    cudaDeviceReset();

    return 0;
}

#endif
