#include <cuda_runtime.h>
#include "utils.h"

__global__ void calculate_next_generation(const bboard* d_a,
    bboard* d_result,
    const int dim,
    const int dim_board_w,
    const int dim_board_h,
    const size_t pitch,
    const int remaining_cells_w,
    const int remaining_cells_h
    )
{
  const int major_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;  // row
  const int major_j = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;  // col
  if (major_i >= dim_board_h) return;
  if (major_j >= dim_board_w) return;

  bboard neighbors[9];
  {
    const int major_l = (major_j - 1 + dim_board_w) % dim_board_w;
    const int major_r = (major_j + 1) % dim_board_w;
    const int major_t = (major_i - 1 + dim_board_h) % dim_board_h;
    const int major_b = (major_i + 1) % dim_board_h;
    bboard* row_c = (bboard*)((char*)d_a + major_i * pitch);
    bboard* row_t = (bboard*)((char*)d_a + major_t * pitch);
    bboard* row_b = (bboard*)((char*)d_a + major_b * pitch);
    neighbors[0] = row_c[major_j];
    neighbors[1] = row_c[major_l];
    neighbors[2] = row_c[major_r];
    neighbors[3] = row_t[major_j];
    neighbors[4] = row_t[major_l];
    neighbors[5] = row_t[major_r];
    neighbors[6] = row_b[major_j];
    neighbors[7] = row_b[major_l];
    neighbors[8] = row_b[major_r];
  }

  const bool is_edge_l = (major_j == 0);
  const bool is_edge_r = (major_j == dim_board_w - 1);
  const char bring_right = WIDTH - 1 - __mul24(remaining_cells_w, is_edge_r);
  const char bring_left = WIDTH - 1 - __mul24(remaining_cells_w, is_edge_l);
  const bboard mask = (bboard)-1 >> __mul24(remaining_cells_w, is_edge_r);

  neighbors[4] = (neighbors[3] << 1) | (neighbors[4] >> bring_left);
  neighbors[5] = (neighbors[3] >> 1) | (neighbors[5] << bring_right);
  neighbors[1] = (neighbors[0] << 1) | (neighbors[1] >> bring_left);
  neighbors[2] = (neighbors[0] >> 1) | (neighbors[2] << bring_right);
  neighbors[7] = (neighbors[6] << 1) | (neighbors[7] >> bring_left);
  neighbors[8] = (neighbors[6] >> 1) | (neighbors[8] << bring_right);

  bboard A[4], A_h[3];

  A[0] = neighbors[1] ^ neighbors[2];
  A[1] = neighbors[1] & neighbors[2];

  A_h[0] = A[0] ^ neighbors[3];
  A_h[1] = A[1] ^ (A[0] & neighbors[3]);

  A[0] = A_h[0] ^ neighbors[4];
  A[1] = A_h[1] ^ (A_h[0] & neighbors[4]);
  A[2] = A_h[1] & A_h[0] & neighbors[4];

  A_h[0] = A[0] ^ neighbors[5];
  A_h[1] = A[1] ^ (A[0] & neighbors[5]);
  A_h[2] = A[2] ^ (A[1] & A[0] & neighbors[5]);

  A[0] = A_h[0] ^ neighbors[6];
  A[1] = A_h[1] ^ (A_h[0] & neighbors[6]);
  A[2] = A_h[2] ^ (A_h[1] & A_h[0] & neighbors[6]);

  A_h[0] = A[0] ^ neighbors[7];
  A_h[1] = A[1] ^ (A[0] & neighbors[7]);
  A_h[2] = A[2] ^ (A[1] & A[0] & neighbors[7]);

  A[0] = A_h[0] ^ neighbors[8];
  A[1] = A_h[1] ^ (A_h[0] & neighbors[8]);
  A[2] = A_h[2] ^ (A_h[1] & A_h[0] & neighbors[8]);
  A[3] = A_h[2] & A_h[1] & A_h[0] & neighbors[8];

  bboard* row_result = (bboard*)((char*)d_result + major_i * pitch);
  row_result[major_j] = ~A[3] & ~A[2] & A[1] & ((~A[0] & neighbors[0]) | A[0]) & mask;
}
