#include <stdint.h>
#include <cuda_runtime.h>
#include "utils.h"

__global__
void convert_from_tiled(int* d_table, const bboard* d_a, const size_t dim, const size_t dim_board, const size_t pitch) {
    UNUSED(dim_board);

    // coordinates in the tiled table.
    const int major_i = blockIdx.y * blockDim.y + threadIdx.y;
    const int major_j = blockIdx.x * blockDim.x + threadIdx.x;
    if (major_i * WIDTH >= dim) return;
    if (major_j * WIDTH >= dim) return;

    const bboard* row_a = (bboard*)((char*)d_a + major_j * pitch);
    const bboard value = row_a[major_i];

    for (int board_i = 0; board_i < WIDTH; board_i++){
        const int real_i = major_i * WIDTH + board_i;
        if (real_i >= dim) break;
        for (int board_j = 0; board_j < WIDTH; board_j++){
            const int real_j = major_j * WIDTH + board_j;
            if (real_j >= dim) break;
            const int real_idx = real_i * dim + real_j;
            d_table[real_idx] = BOARD_IS_SET(value, board_i, board_j);
        }
    }
}

__global__
void convert_to_tiled(const int* d_table, bboard* d_a, const size_t dim, const size_t dim_board, const size_t pitch) {
    UNUSED(dim_board);

    const int major_i = blockIdx.y * blockDim.y + threadIdx.y;
    const int major_j = blockIdx.x * blockDim.x + threadIdx.x;
    if (major_i * WIDTH >= dim) return;
    if (major_j * WIDTH >= dim) return;


    bboard* row_a = (bboard*)((char*)d_a + major_j * pitch);
    bboard value = 0;

    for (int board_i = 0; board_i < WIDTH; board_i++){
        const int real_i = major_i * WIDTH + board_i;
        if (real_i >= dim) break;
        for (int board_j = 0; board_j < WIDTH; board_j++){
            const int real_j = major_j * WIDTH + board_j;
            if (real_j >= dim) break;
            const int real_idx = real_i * dim + real_j;
            if (d_table[real_idx]) {
                SET_BOARD(value, board_i, board_j);
            }
        }
    }
    row_a[major_i] = value;
}
