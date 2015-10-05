#include "utils.h"
#include <cuda_runtime.h>

#define T_I 0
#define C_I 1
#define B_I 2
#define L_J 0
#define C_J 1
#define R_J 2

#include <stdio.h>

__device__
ext_bboard bboard_to_ext(bboard val, int m_i, int m_j) {
    ext_bboard ext = 0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (BOARD_IS_SET(val, i, j)) EXT_SET_BOARD(ext, i, j);
        }
    }
    return ext << (EXT_POS(m_i, m_j));
}

__device__
bboard reverse(bboard x) {
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return ((x >> 16) | (x << 16));

    // inline assembly way:
    //    bboard res;
    //    asm("brev.b32 %0, %1;" : "=r"(res) : "r"(x));
    //    return res;
}

__device__
bboard ext_to_bboard(ext_bboard val) {
    bboard res = 0;
    for (int i = 1; i < EXT_HEIGHT - 1; i++) {
        for (int j = 1; j < EXT_WIDTH - 1; j++) {
            if (EXT_BOARD_IS_SET(val, i, j)) SET_BOARD(res, i - 1, j - 1);
        }
    }
    return res;
}

__device__
ext_bboard gol(ext_bboard cell) {
    const ext_bboard L1 = cell >> 1;
    const ext_bboard L2 = cell << 1;
    const ext_bboard L3 = cell << EXT_WIDTH;
    const ext_bboard L4 = cell >> EXT_WIDTH;
    const ext_bboard L5 = cell << (EXT_WIDTH + 1);
    const ext_bboard L6 = cell >> (EXT_WIDTH + 1);
    const ext_bboard L7 = cell << (EXT_WIDTH - 1);
    const ext_bboard L8 = cell >> (EXT_WIDTH - 1);
    ext_bboard S0, S1, S2, S3;
    S0 = S1 = S2 = S3 = 0;

    S0 = ~(L1 | L2);
    S1 = L1 ^ L2;
    S2 = L1 & L2;

    S3 = L3 & S2;
    S2 = (S2 & ~L3) | (S1 & L3);
    S1 = (S1 & ~L3) | (S0 & L3);
    S0 = S0 & ~L3;

    S3 = (S3 & ~L4) | (S2 & L4);
    S2 = (S2 & ~L4) | (S1 & L4);
    S1 = (S1 & ~L4) | (S0 & L4);
    S0 = S0 & ~L4;

    S3 = (S3 & ~L5) | (S2 & L5);
    S2 = (S2 & ~L5) | (S1 & L5);
    S1 = (S1 & ~L5) | (S0 & L5);
    S0 = S0 & ~L5;

    S3 = (S3 & ~L6) | (S2 & L6);
    S2 = (S2 & ~L6) | (S1 & L6);
    S1 = (S1 & ~L6) | (S0 & L6);
    S0 = S0 & ~L6;

    S3 = (S3 & ~L7) | (S2 & L7);
    S2 = (S2 & ~L7) | (S1 & L7);
    S1 = (S1 & ~L7) | (S0 & L7);
    S0 = S0 & ~L7;

    S3 = (S3 & ~L8) | (S2 & L8);
    S2 = (S2 & ~L8) | (S1 & L8);

    return (((S2 & cell) | S3));
}


#define TPOS(X, Y, DIM)((X) * (DIM) + (Y))

__global__
void calculate_next_generation_no_rem(const bboard* d_a,
                                      bboard* d_result,
                                      const int dim,
                                      const int dim_board_w,
                                      const int dim_board_h,
                                      const size_t pitch
                                     ) {

    const int major_j = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;  // col
    const int major_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;  // row

    if ((__mul24(major_j, WIDTH) >= dim) || (__mul24(major_i, HEIGHT) >= dim)) return;

    bboard neighbors[3][3];
    {
        const int major_l = (major_j - 1 + dim_board_w) % dim_board_w;
        const int major_r = (major_j + 1) % dim_board_w;
        const int major_t = (major_i - 1 + dim_board_h) % dim_board_h;
        const int major_b = (major_i + 1) % dim_board_h;

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

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

        neighbors[C_I][C_J] = tiles[TPOS(tx + 1, ty + 1, by)];
        neighbors[C_I][L_J] = tiles[TPOS(tx + 1, ty, by)];
        neighbors[C_I][R_J] = tiles[TPOS(tx + 1, ty + 2, by)];
        neighbors[T_I][C_J] = tiles[TPOS(tx, ty + 1, by)];
        neighbors[T_I][L_J] = tiles[TPOS(tx, ty, by)];
        neighbors[T_I][R_J] = tiles[TPOS(tx, ty + 2, by)];
        neighbors[B_I][C_J] = tiles[TPOS(tx + 2, ty + 1, by)];
        neighbors[B_I][L_J] = tiles[TPOS(tx + 2, ty, by)];
        neighbors[B_I][R_J] = tiles[TPOS(tx + 2, ty + 2, by)];


        neighbors[C_I][L_J] &= BBOARD_RIGHT_COL_MASK;
        neighbors[C_I][R_J] &= BBOARD_LEFT_COL_MASK;
        neighbors[T_I][C_J] &= BBOARD_BOTTOM_ROW_MASK;
        neighbors[T_I][L_J] &= BBOARD_BOTTOM_ROW_MASK & BBOARD_RIGHT_COL_MASK;
        neighbors[T_I][R_J] &= BBOARD_BOTTOM_ROW_MASK & BBOARD_LEFT_COL_MASK;
        neighbors[B_I][C_J] &= BBOARD_UPPER_ROW_MASK;
        neighbors[B_I][L_J] &= BBOARD_UPPER_ROW_MASK & BBOARD_RIGHT_COL_MASK;
        neighbors[B_I][R_J] &= BBOARD_UPPER_ROW_MASK & BBOARD_LEFT_COL_MASK;

        neighbors[C_I][L_J] = (neighbors[C_I][L_J]) >> (WIDTH - 1);
        neighbors[C_I][R_J] = (neighbors[C_I][R_J]) << (WIDTH - 1);
        neighbors[T_I][C_J] = (neighbors[T_I][C_J]) >> ((HEIGHT - 1) * WIDTH);
        neighbors[T_I][L_J] = reverse(neighbors[T_I][L_J]); // corner
        neighbors[T_I][R_J] = reverse(neighbors[T_I][R_J]); // corner
        neighbors[B_I][C_J] = neighbors[B_I][C_J] << ((HEIGHT - 1) * WIDTH);
        neighbors[B_I][L_J] = reverse(neighbors[B_I][L_J]); // corner
        neighbors[B_I][R_J] = reverse(neighbors[B_I][R_J]); // corner
    }

    ext_bboard res = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            res |= bboard_to_ext(neighbors[i][j], i, j);
        }
    }

    res = gol(res);

    bboard* row_result = (bboard*)((char*)d_result + major_i * pitch);
    row_result[major_j] = ext_to_bboard(res);
}
