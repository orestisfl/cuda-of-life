#include "../utils.h"
#include <cuda_runtime.h>

#define T_I 0
#define C_I 1
#define B_I 2
#define L_J 0
#define C_J 1
#define R_J 2

#include <stdio.h>
__device__
void board_print(bboard val) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            CUDA_PRINT("%s"ANSI_COLOR_RESET,
                       BOARD_IS_SET(val, i, j) ? ANSI_COLOR_BLUE"1 " : ANSI_COLOR_RED"0 ");
        }
        CUDA_PRINT("\n");
    }
    CUDA_PRINT("\n");
}

__device__
void ext_board_print(ext_bboard val) {
    for (int i = 0; i < EXT_HEIGHT; i++) {
        for (int j = 0; j < EXT_WIDTH; j++) {
            CUDA_PRINT("%s"ANSI_COLOR_RESET,
                       EXT_BOARD_IS_SET(val, i, j) ? ANSI_COLOR_BLUE"1 " : ANSI_COLOR_RED"0 ");
        }
        CUDA_PRINT("\n");
    }
    CUDA_PRINT("\n");
}

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

// TODO: rename, optimize
__device__
ext_bboard gol(ext_bboard cell) {
    ext_bboard L1 = cell >> 1;
    ext_bboard L2 = cell << 1;
    ext_bboard L3 = cell << EXT_WIDTH;
    ext_bboard L4 = cell >> EXT_WIDTH;
    ext_bboard L5 = cell << (EXT_WIDTH + 1);
    ext_bboard L6 = cell >> (EXT_WIDTH + 1);
    ext_bboard L7 = cell << (EXT_WIDTH - 1);
    ext_bboard L8 = cell >> (EXT_WIDTH - 1);
    ext_bboard S0 , S1 , S2 , S3 , S4 , S5 , S6 , S7;
    S0 = S1 = S2 = S3 = S4 = S5 = S6 = S7 = 0;

    S0 = ~(L1 | L2);
    S1 = L1 ^ L2;
    S2 = L1 & L2;

    S3 = L3 & S2;
    S2 = (S2 & ~L3) | (S1 & L3);
    S1 = (S1 & ~L3) | (S0 & L3);
    S0 = S0 & ~L3;

    S4 = S3 & L4;
    S3 = (S3 & ~L4) | (S2 & L4);
    S2 = (S2 & ~L4) | (S1 & L4);
    S1 = (S1 & ~L4) | (S0 & L4);
    S0 = S0 & ~L4;

    S5 = S4 & L5;
    S4 = (S4 & ~L5) | (S3 & L5);
    S3 = (S3 & ~L5) | (S2 & L5);
    S2 = (S2 & ~L5) | (S1 & L5);
    S1 = (S1 & ~L5) | (S0 & L5);
    S0 = S0 & ~L5;

    S6 = S5 & L6;
    S5 = (S5 & ~L6) | (S4 & L6);
    S4 = (S4 & ~L6) | (S3 & L6);
    S3 = (S3 & ~L6) | (S2 & L6);
    S2 = (S2 & ~L6) | (S1 & L6);
    S1 = (S1 & ~L6) | (S0 & L6);
    S0 = S0 & ~L6;

    S7 = S6 & L7;
    S6 = (S6 & ~L7) | (S5 & L7);
    S5 = (S5 & ~L7) | (S4 & L7);
    S4 = (S4 & ~L7) | (S3 & L7);
    S3 = (S3 & ~L7) | (S2 & L7);
    S2 = (S2 & ~L7) | (S1 & L7);
    S1 = (S1 & ~L7) | (S0 & L7);
    S0 = S0 & ~L7;

    S7 = (S7 & ~L8) | (S6 & L8);
    S6 = (S6 & ~L8) | (S5 & L8);
    S5 = (S5 & ~L8) | (S4 & L8);
    S4 = (S4 & ~L8) | (S3 & L8);
    S3 = (S3 & ~L8) | (S2 & L8);
    S2 = (S2 & ~L8) | (S1 & L8);
    // ~ S1 = (S1 & ~L8) | (S0 & L8);
    // ~ S0 = S0 & ~L8;

    // ~ board_print(S2 & cell);
    // ~ board_print(S3);
    // ~ return (((S2 & cell) | S3) & 8289792);
    return (((S2 & cell) | S3));
}


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
        bboard* row_c = (bboard*)((char*)d_a + major_i * pitch);
        bboard* row_t = (bboard*)((char*)d_a + major_t* pitch);
        bboard* row_b = (bboard*)((char*)d_a + major_b * pitch);
        neighbors[C_I][C_J] = row_c[major_j];
        neighbors[C_I][L_J] = row_c[major_l] & BBOARD_RIGHT_COL_MASK;
        neighbors[C_I][R_J] = row_c[major_r] & BBOARD_LEFT_COL_MASK;
        neighbors[T_I][C_J] = row_t[major_j] & BBOARD_BOTTOM_ROW_MASK;
        neighbors[T_I][L_J] = row_t[major_l] & BBOARD_BOTTOM_ROW_MASK & BBOARD_RIGHT_COL_MASK;
        neighbors[T_I][R_J] = row_t[major_r] & BBOARD_BOTTOM_ROW_MASK & BBOARD_LEFT_COL_MASK;
        neighbors[B_I][C_J] = row_b[major_j] & BBOARD_UPPER_ROW_MASK;
        neighbors[B_I][L_J] = row_b[major_l] & BBOARD_UPPER_ROW_MASK & BBOARD_RIGHT_COL_MASK;
        neighbors[B_I][R_J] = row_b[major_r] & BBOARD_UPPER_ROW_MASK & BBOARD_LEFT_COL_MASK;

        neighbors[C_I][L_J] = (neighbors[C_I][L_J]) >> (WIDTH - 1);
        neighbors[C_I][R_J] = (neighbors[C_I][R_J]) << (WIDTH - 1);
        neighbors[T_I][C_J] = (neighbors[T_I][C_J]) >> ((HEIGHT - 1) * WIDTH);
        neighbors[T_I][L_J] = reverse(neighbors[T_I][L_J]); // corner
        neighbors[T_I][R_J] = reverse(neighbors[T_I][R_J]); // corner
        neighbors[B_I][C_J] = neighbors[B_I][C_J] << ((HEIGHT - 1) * WIDTH);
        neighbors[B_I][L_J] = reverse(neighbors[B_I][L_J]); // corner
        neighbors[B_I][R_J] = reverse(neighbors[B_I][R_J]); // corner
    }

    ext_bboard trans = 0;
    // TODO: unroll?
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ext_bboard tmp = bboard_to_ext(neighbors[i][j], i, j);
//            if (threadIdx.x == 0) CUDA_PRINT("from:\n");
//            if (threadIdx.x == 0) board_print(neighbors[i][j]);
//            if (threadIdx.x == 0) CUDA_PRINT("to:\n");
//            if (threadIdx.x == 0) ext_board_print(tmp);
            trans |= tmp;
        }
    }

    ext_bboard res = gol(trans);
    // unessecary if we keep ext_to_board() like that
    res &= EXT_BBOARD_CENTER_MASK;

    bboard value = ext_to_bboard(res);

//    if (threadIdx.x == 0) board_print(neighbors[C_I][C_J]);
//    if (threadIdx.x == 0) ext_board_print(trans);
//    if (threadIdx.x == 0) ext_board_print(res);
//    if (threadIdx.x == 0) board_print(value);

    bboard* row_result = (bboard*)((char*)d_result + major_i * pitch);
    row_result[major_j] = value;
}
