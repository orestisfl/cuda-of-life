#define j 0
right_j = j + 1;
right_n = C_J;
left_j = WIDTH - 1 - remaining_cells_w * is_edge_l;
this_cell = BOARD_IS_SET(neighbors[C_I][C_J], i, j);
first_cells = BOARD_IS_SET(neighbors[up_n][L_J], up_i, left_j)
              + BOARD_IS_SET(neighbors[C_I][L_J], i, left_j)
              + BOARD_IS_SET(neighbors[down_n][L_J], down_i, left_j);
second_cells = BOARD_IS_SET(neighbors[up_n][C_J], up_i, j)
               + this_cell
               + BOARD_IS_SET(neighbors[down_n][C_J], down_i, j);
#include "kafrila2.c"
#undef j

for (int j = 1; j < limit_j - 1; j++) {
    this_cell = BOARD_IS_SET(neighbors[C_I][C_J], i, j);

    right_j = j + 1;
    right_n = C_J;

    #include "kafrila2.c"
}

#define j (limit_j - 1)
this_cell = BOARD_IS_SET(neighbors[C_I][C_J], i, j);
right_j = 0;
right_n = R_J;
#include "kafrila2.c"
#undef j
