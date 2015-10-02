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
  }
  else {
    bx = blockDim.x + 2;
  }
  if (blockIdx.y == gridDim.y - 1) {
    by = dim_board_w - blockIdx.y * blockDim.y + 2;
  }
  else {
    by = blockDim.y + 2;
  }

  int major_t = (major_i - 1 + dim_board_h) % dim_board_h;
  int major_b = (major_i + 1) % dim_board_h;
  int major_l = (major_j - 1 + dim_board_w) % dim_board_w;
  int major_r = (major_j + 1) % dim_board_w;
  bboard* top_row = (bboard*)((char*)d_a + major_t * pitch);
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
#pragma unroll
  for (char i = 0; i < limit_i; i++) {
    char up_i, up_n, down_i, down_n;
    char first_cells, second_cells;
    char alive_cells, this_cell;

    if (i == 0) {
      up_i = HEIGHT - 1 - remaining_cells_h * is_edge_u;
      up_n = T_I;
    } else {
      up_i = i - 1;
      up_n = C_I;
    }
    if (i == limit_i - 1) {
      down_i = 0;
      down_n = B_I;
    } else {
      down_i = i + 1;
      down_n = C_I;
    }

#pragma unroll
    for (int j = 0; j < limit_j; j++) {
      this_cell = BOARD_IS_SET(neighbors[C_I][C_J], i, j);
      char right_j, right_n;

      if (j == 0) {
        int left_j = WIDTH - 1 - remaining_cells_w * is_edge_l;
        first_cells = BOARD_IS_SET(neighbors[up_n][L_J], up_i, left_j)
          + BOARD_IS_SET(neighbors[C_I][L_J], i, left_j)
          + BOARD_IS_SET(neighbors[down_n][L_J], down_i, left_j);
        second_cells = BOARD_IS_SET(neighbors[up_n][C_J], up_i, j)
          + this_cell
          + BOARD_IS_SET(neighbors[down_n][C_J], down_i, j);
      }
      if (j == limit_j - 1) {
        right_j = 0;
        right_n = R_J;
      } else {
        right_j = j + 1;
        right_n = C_J;
      }

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

      bool set = (alive_cells == 3) || (alive_cells == 2 && this_cell);

      if (set) {
        SET_BOARD(value, i, j);
      }
    }

  }

  bboard* row_result = (bboard*)((char*)d_result + major_i * pitch);
  row_result[major_j] = value;
}
