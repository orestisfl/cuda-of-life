#include <cuda_runtime.h>
#include "utils.h"

#define T_I 0
#define C_I 1
#define B_I 2
#define L_J 0
#define C_J 1
#define R_J 2

__global__ void calculate_next_generation(const bboard* d_a, bboard* d_result, const int dim,
    const int dim_board, const size_t pitch) {
  const int major_i = blockIdx.y * blockDim.y + threadIdx.y;  // row
  const int major_j = blockIdx.x * blockDim.x + threadIdx.x;  // col
  if (major_i * WIDTH >= dim) return;
  if (major_j * WIDTH >= dim) return;

  const int major_l = (major_j - 1 + dim_board) % dim_board;
  const int major_r = (major_j + 1) % dim_board;
  const int major_t = (major_i - 1 + dim_board) % dim_board;
  const int major_b = (major_i + 1) % dim_board;

  bboard* row_c = (bboard*)((char*)d_a + major_i * pitch);
  bboard* row_t = (bboard*)((char*)d_a + major_t* pitch);
  bboard* row_b = (bboard*)((char*)d_a + major_b * pitch);

  bboard neighbors[3][3];
  neighbors[C_I][C_J] = row_c[major_j];
  neighbors[C_I][L_J] = row_c[major_l];
  neighbors[C_I][R_J] = row_c[major_r];
  neighbors[T_I][C_J] = row_t[major_j];
  neighbors[T_I][L_J] = row_t[major_l];
  neighbors[T_I][R_J] = row_t[major_r];
  neighbors[B_I][C_J] = row_b[major_j];
  neighbors[B_I][L_J] = row_b[major_l];
  neighbors[B_I][R_J] = row_b[major_r];

  //TODO: move global?
  const int remaining_dim = gridDim.x * blockDim.x * WIDTH - dim;
  //    const int remaining_blocks = remaining_dim / WIDTH; // for shared??
  const int remaining_cells = remaining_dim % WIDTH;
  //    assert(remaining_dim >= 0);
  //    assert(remaining_blocks >= 0);
  //    assert(remaining_cells >= 0);

  const bool is_edge_r = (major_j == dim_board - 1);
  const bool is_edge_d = (major_i == dim_board - 1);
  const bool is_edge_u = (major_i == 0);
  const bool is_edge_l = (major_j == 0);

  const int limit_i = WIDTH - remaining_cells * is_edge_d;
  const int limit_j = WIDTH - remaining_cells * is_edge_r;

  if (is_edge_d) CUDA_PRINT("%d %d is_edge_d\n", major_i, major_j);
  if (is_edge_r) CUDA_PRINT("%d %d is_edge_r\n", major_i, major_j);
  if (is_edge_u) CUDA_PRINT("%d %d is_edge_u\n", major_i, major_j);
  if (is_edge_l) CUDA_PRINT("%d %d is_edge_l\n", major_i, major_j);

  bboard value = 0;
  int up_i, up_n, down_i, down_n;
  int right_j, right_n;
  uint first_cells, second_cells;
  uint alive_cells, this_cell;
  for (int i = 0; i < limit_i; i++) {

    if (i == 0) {
      up_i = WIDTH - 1 - remaining_cells * is_edge_u;
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

    for (int j = 0; j < limit_j; j++) {

      this_cell = BOARD_IS_SET(neighbors[C_I][C_J], i, j);

      if (j == 0) {
        int left_j = WIDTH - 1 - remaining_cells * is_edge_l;
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
      }
      else {
        alive_cells = first_cells;
        first_cells = BOARD_IS_SET(neighbors[up_n][right_n], up_i, right_j)
          + BOARD_IS_SET(neighbors[C_I][right_n], i, right_j)
          + BOARD_IS_SET(neighbors[down_n][right_n], down_i, right_j);
        alive_cells += first_cells;
        alive_cells += second_cells - this_cell;
      }

      const bool set = (alive_cells == 3) || (alive_cells == 2 && this_cell);
      if (set) {
        SET_BOARD(value, i, j);
      }
    }

  }

  bboard* row_result = (bboard*)((char*)d_result + major_i * pitch);
  row_result[major_j] = value;
}
