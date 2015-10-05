#include <cuda_runtime.h>
#include "utils.h"

#define TPOS(X, Y, DIM)((X) * (DIM) + (Y))
#define T_I 0
#define C_I 1
#define B_I 2
#define L_J 0
#define C_J 1
#define R_J 2

__global__ void calculate_next_generation(const bboard* d_a,
                                          bboard* d_result,
                                          const int dim,
                                          const int dim_board_w,
                                          const int dim_board_h,
                                          const size_t pitch,
                                          const int remaining_cells_w,
                                          const int remaining_cells_h
                                         ) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int major_i = __mul24(blockIdx.x, blockDim.x) + tx;  // row
    const int major_j = __mul24(blockIdx.y, blockDim.y) + ty;  // col

    if (major_i >= dim_board_h) return;
    if (major_j >= dim_board_w) return;

    extern __shared__ bboard tiles[];

    int bx, by;
    if (blockIdx.x == gridDim.x - 1) {
        bx = dim_board_h - blockIdx.x * blockDim.x + 2;
    } else {
        bx = blockDim.x + 2;
    }
    if (blockIdx.y == gridDim.y - 1) {
        by = dim_board_w - blockIdx.y * blockDim.y + 2;
    } else {
        by = blockDim.y + 2;
    }

    int major_t = (major_i - 1 + dim_board_h) % dim_board_h;
    int major_b = (major_i + 1) % dim_board_h;
    int major_l = (major_j - 1 + dim_board_w) % dim_board_w;
    int major_r = (major_j + 1) % dim_board_w;
    bboard* top_row = (bboard*)((char*)d_a + major_t* pitch);
    bboard* row = (bboard*)((char*)d_a + major_i * pitch);
    bboard* bot_row = (bboard*)((char*)d_a + major_b * pitch);

    tiles[TPOS(tx + 1, ty + 1, by)] = row[major_j];

    if (ty == 0) {
        //is in the left edge of the block. keep row
        tiles[TPOS(tx + 1, 0, by)] = row[major_l];

        if (tx == 0) {
            //top left corner!
            tiles[TPOS(0, 0, by)] = top_row[major_l];
        }

        if (tx == bx - 3) {
            //bottom left corner
            tiles[TPOS(bx - 1, 0, by)] = bot_row[major_l];
        }
    }

    if (ty == by - 3) {
        //is on the right edge
        tiles[TPOS(tx + 1, by - 1, by)] = row[major_r];

        if (tx == 0) {
            // top right corner
            tiles[TPOS(0, by - 1, by)] = top_row[major_r];
        }

        if (tx == bx - 3) {
            // bottom right corner
            tiles[TPOS(bx - 1, by - 1, by)] = bot_row[major_r];
        }
    }

    if (tx == 0) {
        //is on the upper edge of the block. keep col
        tiles[TPOS(0, ty + 1, by)] = top_row[major_j];
    }

    if (tx == bx - 3) {
        //is on the bottom edge
        tiles[TPOS(bx - 1, ty + 1, by)] = bot_row[major_j];
    }

    __syncthreads();

    bboard neighbors[3][3];
    neighbors[C_I][C_J] = tiles[TPOS(tx + 1, ty + 1, by)];
    neighbors[C_I][L_J] = tiles[TPOS(tx + 1, ty, by)];
    neighbors[C_I][R_J] = tiles[TPOS(tx + 1, ty + 2, by)];
    neighbors[T_I][C_J] = tiles[TPOS(tx, ty + 1, by)];
    neighbors[T_I][L_J] = tiles[TPOS(tx, ty, by)];
    neighbors[T_I][R_J] = tiles[TPOS(tx, ty + 2, by)];
    neighbors[B_I][C_J] = tiles[TPOS(tx + 2, ty + 1, by)];
    neighbors[B_I][L_J] = tiles[TPOS(tx + 2, ty, by)];
    neighbors[B_I][R_J] = tiles[TPOS(tx + 2, ty + 2, by)];

    const bool is_edge_r = (major_j == dim_board_w - 1);
    const bool is_edge_d = (major_i == dim_board_h - 1);
    const bool is_edge_u = (major_i == 0);
    const bool is_edge_l = (major_j == 0);

    const char limit_i = HEIGHT - __mul24(remaining_cells_h, is_edge_d);
    const char limit_j = WIDTH - __mul24(remaining_cells_w, is_edge_r);

    bboard value = 0;
    char first_cells, second_cells;
    char alive_cells, this_cell;
    //    char left_j;
    //    bool set;

#define i 0
#define up_i (HEIGHT - 1 - remaining_cells_h * is_edge_u)
#define up_n T_I
#define down_i (i + 1)
#define down_n (C_I)
#include "code_includes/kafrila.c"
#undef i

    for (char i = 1; i < limit_i - 1; i++) {
#define up_i (i - 1)
#define up_n C_I
#define down_i (i + 1)
#define down_n C_I
#include "code_includes/kafrila.c"
    }

#define i (limit_i - 1)
#define up_i (i - 1)
#define up_n C_I
#define down_i 0
#define down_n B_I
#include "code_includes/kafrila.c"
#undef i
    bboard* row_result = (bboard*)((char*)d_result + major_i * pitch);
    row_result[major_j] = value;
}
