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

  bboard a[9];
  {
    const int major_l = (major_j - 1 + dim_board_w) % dim_board_w;
    const int major_r = (major_j + 1) % dim_board_w;
    const int major_t = (major_i - 1 + dim_board_h) % dim_board_h;
    const int major_b = (major_i + 1) % dim_board_h;
    bboard* row_c = (bboard*)((char*)d_a + major_i * pitch);
    bboard* row_t = (bboard*)((char*)d_a + major_t * pitch);
    bboard* row_b = (bboard*)((char*)d_a + major_b * pitch);
    a[0] = row_c[major_j];
    a[1] = row_c[major_l];
    a[2] = row_c[major_r];
    a[3] = row_t[major_j];
    a[4] = row_t[major_l];
    a[5] = row_t[major_r];
    a[6] = row_b[major_j];
    a[7] = row_b[major_l];
    a[8] = row_b[major_r];
  }

  const bool is_edge_l = (major_j == 0);
  const bool is_edge_r = (major_j == dim_board_w - 1);
  const char bring_right = WIDTH - 1 - __mul24(remaining_cells_w, is_edge_r);
  const char bring_left = WIDTH - 1 - __mul24(remaining_cells_w, is_edge_l);
  const bboard mask = (bboard)-1 >> __mul24(remaining_cells_w, is_edge_r);

  a[4] = (a[3] << 1) | (a[4] >> bring_left);
  a[5] = (a[3] >> 1) | (a[5] << bring_right);
  a[1] = (a[0] << 1) | (a[1] >> bring_left);
  a[2] = (a[0] >> 1) | (a[2] << bring_right);
  a[7] = (a[6] << 1) | (a[7] >> bring_left);
  a[8] = (a[6] >> 1) | (a[8] << bring_right);

  bboard A[4], A_h[3];

  A[0] = a[1] ^ a[2];
  A[1] = a[1] & a[2];

  A_h[0] = A[0] ^ a[3];
  A_h[1] = A[1] ^ (A[0] & a[3]);

  A[0] = A_h[0] ^ a[4];
  A[1] = A_h[1] ^ (A_h[0] & a[4]);
  A[2] = A_h[1] & A_h[0] & a[4];

  A_h[0] = A[0] ^ a[5];
  A_h[1] = A[1] ^ (A[0] & a[5]);
  A_h[2] = A[2] ^ (A[1] & A[0] & a[5]);

  A[0] = A_h[0] ^ a[6];
  A[1] = A_h[1] ^ (A_h[0] & a[6]);
  A[2] = A_h[2] ^ (A_h[1] & A_h[0] & a[6]);

  A_h[0] = A[0] ^ a[7];
  A_h[1] = A[1] ^ (A[0] & a[7]);
  A_h[2] = A[2] ^ (A[1] & A[0] & a[7]);

  A[0] = A_h[0] ^ a[8];
  A[1] = A_h[1] ^ (A_h[0] & a[8]);
  A[2] = A_h[2] ^ (A_h[1] & A_h[0] & a[8]);
  A[3] = A_h[2] & A_h[1] & A_h[0] & a[8];

  bboard* row_result = (bboard*)((char*)d_result + major_i * pitch);
  row_result[major_j] = ~A[3] & ~A[2] & A[1] & ((~A[0] & a[0]) | A[0]) & mask;
}
