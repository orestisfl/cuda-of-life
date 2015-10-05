if (j & 1u) {
    alive_cells = second_cells;
    second_cells = BOARD_IS_SET(neighbors[up_n][right_n], up_i, right_j)
                   + BOARD_IS_SET(neighbors[C_I][right_n], i, right_j)
                   + BOARD_IS_SET(neighbors[down_n][right_n], down_i, right_j);
    alive_cells += second_cells;
    alive_cells += first_cells - this_cell;
} else {
    alive_cells = first_cells;
    first_cells = BOARD_IS_SET(neighbors[up_n][right_n], up_i, right_j)
                  + BOARD_IS_SET(neighbors[C_I][right_n], i, right_j)
                  + BOARD_IS_SET(neighbors[down_n][right_n], down_i, right_j);
    alive_cells += first_cells;
    alive_cells += second_cells - this_cell;
}

if ((alive_cells == 3) || (alive_cells == 2 && this_cell)) {
    SET_BOARD(value, i, j);
}

#undef right_j
#undef right_n
