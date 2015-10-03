#include <cuda_runtime.h>
#include "utils.h"

#define T_I 0
#define C_I 1
#define B_I 2
#define L_J 0
#define C_J 1
#define R_J 2

#ifdef NO_REMAINDERS
#define remaining_cells_h 0
#define remaining_cells_w 0

__device__
bboard calculate_midle(bboard value) {
    return value;
}

__global__
void calculate_next_generation_no_rem(const bboard* d_a,
                                      bboard* d_result,
                                      const int dim,
                                      const int dim_board_w,
                                      const int dim_board_h,
                                      const size_t pitch
                                     ) {
#else
__global__
void calculate_next_generation(const bboard* d_a,
                               bboard* d_result,
                               const int dim,
                               const int dim_board_w,
                               const int dim_board_h,
                               const size_t pitch,
                               const int remaining_cells_w,
                               const int remaining_cells_h
                              ) {
#endif

    const int major_j = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;  // col
    const int major_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;  // row

    if ((__mul24(major_j, WIDTH) >= dim) || (__mul24(major_i, HEIGHT) >= dim)) return;

    bboard neighbors[3][3];
    {
        const int major_l = (major_j - 1 + dim_board_w) % dim_board_w;
        const int major_r = (major_j + 1) % dim_board_w;
        const int major_t = (major_i - 1 + dim_board_h) % dim_board_h;
        const int major_b = (major_i + 1) % dim_board_h;
        bboard* row_c = (bboard*)((char*)d_a + major_i * pitch);
        bboard* row_t = (bboard*)((char*)d_a + major_t* pitch);
        bboard* row_b = (bboard*)((char*)d_a + major_b * pitch);
        neighbors[C_I][C_J] = row_c[major_j];
        neighbors[C_I][L_J] = row_c[major_l];
        neighbors[C_I][R_J] = row_c[major_r];
        neighbors[T_I][C_J] = row_t[major_j];
        neighbors[T_I][L_J] = row_t[major_l];
        neighbors[T_I][R_J] = row_t[major_r];
        neighbors[B_I][C_J] = row_b[major_j];
        neighbors[B_I][L_J] = row_b[major_l];
        neighbors[B_I][R_J] = row_b[major_r];
    }

    #ifdef NO_REMAINDERS
#define limit_i HEIGHT
#define limit_j WIDTH
#define is_edge_u false
#define is_edge_l false
    #else
    const bool is_edge_r = (major_j == dim_board_w - 1);
    const bool is_edge_d = (major_i == dim_board_h - 1);
    const char limit_i = HEIGHT - __mul24(remaining_cells_h, is_edge_d);
    const char limit_j = WIDTH - __mul24(remaining_cells_w, is_edge_r);
    const bool is_edge_u = (major_i == 0);
    const bool is_edge_l = (major_j == 0);
    #endif

    bboard value = 0;
    char first_cells, second_cells;
    char alive_cells, this_cell;

#define i 0
#define up_i (HEIGHT - 1 - remaining_cells_h * is_edge_u)
#define up_n T_I
#define down_i (i + 1)
#define down_n (C_I)
#include "kafrila.c"
#undef i


    for (char i = 1; i < limit_i - 1; i++) {
#define up_i (i - 1)
#define up_n C_I
#define down_i (i + 1)
#define down_n C_I
#include "kafrila.c"
    }

#define i (limit_i - 1)
#define up_i (i - 1)
#define up_n C_I
#define down_i 0
#define down_n B_I
#include "kafrila.c"
#undef i
    bboard* row_result = (bboard*)((char*)d_result + major_i * pitch);
    row_result[major_j] = value;
}

#ifdef NO_REMAINDERS
    // clean up
    #undef limit_i
    #undef limit_j
    #undef is_edge_u
    #undef is_edge_l
    #undef remaining_cells_h
    #undef remaining_cells_w
#endif