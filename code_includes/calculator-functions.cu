#include "../utils.h"
#include <cuda_runtime.h>

#define T_I 0
#define C_I 1
#define B_I 2
#define L_J 0
#define C_J 1
#define R_J 2

#ifdef NO_REMAINDERS
#define remaining_cells_h 0
#define remaining_cells_w 0

#include <stdio.h>
// ~ __device__
// ~ void board_print(bboard val) {
    // ~ for (int i = 0; i < HEIGHT; i++) {
        // ~ for (int j = 0; j < WIDTH; j++) {
            // ~ printf("%s"ANSI_COLOR_RESET,
                   // ~ BOARD_IS_SET(val, i, j) ? ANSI_COLOR_BLUE"1 " : ANSI_COLOR_RED"0 ");
        // ~ }
        // ~ printf("\n");
    // ~ }
    // ~ printf("\n");
// ~ }

__device__
static bboard calculate_midle(bboard this_cell) {
    // ~ const bboard L1 = this_cell >> 1;
    // ~ const bboard L2 = this_cell << 1;
    // ~ const bboard L3 = this_cell << WIDTH;
    // ~ const bboard L4 = this_cell >> WIDTH;
    // ~ const bboard L5 = this_cell << (WIDTH + 1);
    // ~ const bboard L6 = this_cell >> (WIDTH + 1);
    // ~ const bboard L7 = this_cell << (WIDTH - 1);
    // ~ const bboard L8 = this_cell >> (WIDTH - 1);
    // ~ bboard S0, S1, S2, S3;

    // ~ S0 = ~(L1 | L2);
    // ~ S1 = L1 ^ L2;
    // ~ S2 = L1 & L2;

    // ~ S3 = L3 & S2;
    // ~ S2 = (S2 & ~L3) | (S1 & L3);
    // ~ S1 = (S1 & ~L3) | (S0 & L3);
    // ~ S0 = S0 & ~L3;

    // ~ S3 = (S3 & ~L4) | (S2 & L4);
    // ~ S2 = (S2 & ~L4) | (S1 & L4);
    // ~ S1 = (S1 & ~L4) | (S0 & L4);
    // ~ S0 = S0 & ~L4;

    // ~ S3 = (S3 & ~L5) | (S2 & L5);
    // ~ S2 = (S2 & ~L5) | (S1 & L5);
    // ~ S1 = (S1 & ~L5) | (S0 & L5);
    // ~ S0 = S0 & ~L5;

    // ~ S3 = (S3 & ~L6) | (S2 & L6);
    // ~ S2 = (S2 & ~L6) | (S1 & L6);
    // ~ S1 = (S1 & ~L6) | (S0 & L6);
    // ~ S0 = S0 & ~L6;

    // ~ S3 = (S3 & ~L7) | (S2 & L7);
    // ~ S2 = (S2 & ~L7) | (S1 & L7);
    // ~ S1 = (S1 & ~L7) | (S0 & L7);

    // ~ S3 = (S3 & ~L8) | (S2 & L8);
    // ~ S2 = (S2 & ~L8) | (S1 & L8);
    bboard L1 = this_cell >> 1;
    bboard L2 = this_cell << 1;
    bboard L3 = this_cell << WIDTH;
    bboard L4 = this_cell >> WIDTH;
    bboard L5 = this_cell << (WIDTH + 1);
    bboard L6 = this_cell >> (WIDTH + 1);
    bboard L7 = this_cell << (WIDTH - 1);
    bboard L8 = this_cell >> (WIDTH - 1);
    bboard S0 , S1 , S2 , S3 , S4 , S5 , S6 , S7 , S8;
    S0 = S1 = S2 = S3 = S4 = S5 = S6 = S7 = S8 = 0;
    // ~ ADD2(S0, S1, L1, L2);

    // ~ board_print(S0);
    // ~ board_print(S1);
    // ~ board_print(L1);
    // ~ board_print(L2);

    S0 = ~(L1 | L2);
    S1 = L1 ^ L2;
    S2 = L1 & L2;
    //Now building on the results above consider the third layer.
    //Notice for S2, the sum is two if the sum of the first two terms
    //was two and the third is zero, or the sum is two if the sum of
    //the first two terms was one and the third term is one.
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
    
    S8 = S7 & L8;
    S7 = (S7 & ~L8) | (S6 & L8);
    S6 = (S6 & ~L8) | (S5 & L8);
    S5 = (S5 & ~L8) | (S4 & L8);
    S4 = (S4 & ~L8) | (S3 & L8);
    S3 = (S3 & ~L8) | (S2 & L8);
    S2 = (S2 & ~L8) | (S1 & L8);

    return ((S2 & this_cell) | S3) & BBOARD_CENTER_MASK;
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

    char first_cells, second_cells;
    char alive_cells, this_cell;

    #ifdef NO_REMAINDERS
    #define this_cc neighbors[C_I][C_J]
    #define this_cl neighbors[C_I][L_J]
    #define this_cr neighbors[C_I][R_J]
    bboard value = calculate_midle(this_cc);
    // ~ if (threadIdx.x == 1) board_print(value);
    // ~ if (threadIdx.x == 1) board_print(this_cc);
    // ~ for (char i = 1; i < limit_i - 1; i++) {

        // ~ char alive_neighbors;
        // ~ bool set;

        // ~ alive_neighbors =
            // ~ BOARD_IS_SET(this_cl, i - 1, WIDTH - 1) + // top left
            // ~ BOARD_IS_SET(this_cc, i - 1, 0) + // top center
            // ~ BOARD_IS_SET(this_cc, i - 1, 1) + // top right
            // ~ BOARD_IS_SET(this_cl, i, WIDTH - 1) + // center left
            // ~ BOARD_IS_SET(this_cc, i, 1) + // center right
            // ~ BOARD_IS_SET(this_cl, i + 1, WIDTH - 1) + // bottom left
            // ~ BOARD_IS_SET(this_cc, i + 1, 0) + // bottom center
            // ~ BOARD_IS_SET(this_cc, i + 1, 1);  // bottom right

        // ~ set = (alive_neighbors == 3) || (alive_neighbors == 2 &&
                                         // ~ BOARD_IS_SET(this_cc, i, 0));
        // ~ if (set) SET_BOARD(value, i, 0);

        // ~ alive_neighbors =
            // ~ BOARD_IS_SET(this_cc, i - 1, limit_j - 2) + // top left
            // ~ BOARD_IS_SET(this_cc, i - 1, limit_j - 1) + // top center
            // ~ BOARD_IS_SET(this_cr, i - 1, 0) + // top right
            // ~ BOARD_IS_SET(this_cc, i, limit_j - 2) + // center left
            // ~ BOARD_IS_SET(this_cr, i, 0) + // center right
            // ~ BOARD_IS_SET(this_cc, i + 1, limit_j - 2) + // bottom left
            // ~ BOARD_IS_SET(this_cc, i + 1, limit_j - 1) + // bottom center
            // ~ BOARD_IS_SET(this_cr, i + 1, 0);  // bottom right

        // ~ set = (alive_neighbors == 3) || (alive_neighbors == 2 &&
                                         // ~ BOARD_IS_SET(this_cc, i, limit_j - 1));
        // ~ if (set) SET_BOARD(value, i, limit_j - 1);
    // ~ }
    // ~ for (char i = 1; i < limit_i - 1; i++){
        // ~ #define up_i (i-1)
        // ~ #define up_n C_I
        // ~ #define down_i (i+1)
        // ~ #define down_n C_I
        // ~ #include "kafrila_jlim.c"
    // ~ }

        for (char i = 1; i < HEIGHT - 2; i++) {
#define up_i (i - 1)
#define up_n C_I
#define down_n C_I
#define down_i (i+1)

#define j 0
#define left_j (WIDTH - 1)
#define left_n L_J
#define right_j (j + 1)
#define right_n C_J
        int alive_neighbors =
            BOARD_IS_SET(neighbors[up_n][left_n], up_i, left_j) + // top left
            BOARD_IS_SET(neighbors[up_n][C_J], up_i, j) + // top center
            BOARD_IS_SET(neighbors[up_n][right_n], up_i, right_j) + // top right
            BOARD_IS_SET(neighbors[C_I][left_n], i, left_j) + // center left
            //                BOARD_IS_SET(neighbors[C_I][C_J], i, j) + // center center
            BOARD_IS_SET(neighbors[C_I][right_n], i, right_j) + // center right
            BOARD_IS_SET(neighbors[down_n][left_n], down_i, left_j) + // bottom left
            BOARD_IS_SET(neighbors[down_n][C_J], down_i, j) + // bottom center
            BOARD_IS_SET(neighbors[down_n][right_n], down_i, right_j);  // bottom right
        bool set = (alive_neighbors == 3) || (alive_neighbors == 2 &&
                                              BOARD_IS_SET(neighbors[C_I][C_J], i, j));
        if (set) SET_BOARD(value, i, j);

#undef left_j
#undef left_n
#undef right_n
#undef right_j
#undef j
#define j (HEIGHT - 1)
#define left_j (j - 1)
#define left_n C_J
#define right_j 0
#define right_n R_J
        alive_neighbors =
            BOARD_IS_SET(neighbors[up_n][left_n], up_i, left_j) + // top left
            BOARD_IS_SET(neighbors[up_n][C_J], up_i, j) + // top center
            BOARD_IS_SET(neighbors[up_n][right_n], up_i, right_j) + // top right
            BOARD_IS_SET(neighbors[C_I][left_n], i, left_j) + // center left
            //                BOARD_IS_SET(neighbors[C_I][C_J], i, j) + // center center
            BOARD_IS_SET(neighbors[C_I][right_n], i, right_j) + // center right
            BOARD_IS_SET(neighbors[down_n][left_n], down_i, left_j) + // bottom left
            BOARD_IS_SET(neighbors[down_n][C_J], down_i, j) + // bottom center
            BOARD_IS_SET(neighbors[down_n][right_n], down_i, right_j);  // bottom right
        set = (alive_neighbors == 3) || (alive_neighbors == 2 &&
                                         BOARD_IS_SET(neighbors[C_I][C_J], i, j));
        if (set) SET_BOARD(value, i, j);
#undef left_j
#undef left_n
#undef right_n
#undef right_j
#undef down_n
#undef down_i
#undef up_i
#undef up_n
#undef j

    }
    #else
    bboard value = 0;
    for (char i = 1; i < limit_i - 1; i++) {
#define up_i (i - 1)
#define up_n C_I
#define down_i (i + 1)
#define down_n C_I
#include "kafrila.c"
    }
    #endif

#define i 0
#define up_i (HEIGHT - 1 - remaining_cells_h * is_edge_u)
#define up_n T_I
#define down_i (i + 1)
#define down_n (C_I)
#include "kafrila.c"
#undef i

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
