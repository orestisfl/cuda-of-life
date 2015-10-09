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
  const int tx = threadIdx.x;
  const int bdx = blockDim.x;
  const int major_i = __mul24(blockIdx.x, bdx) + tx;  // row
  const int major_j = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;  // col

  extern __shared__ bboard tiles[];
  bboard neighbors[9];
  bboard* row_c = (bboard*)((char*)d_a + (major_i % dim_board_h) * pitch);
  int major_t, major_b;
  bboard *row_t, *row_b;

  //!< booleans to check whether thread lies on matrix horizontal edge
  const bool is_edge_l = (major_j == 0);
  const bool is_edge_r = (major_j == dim_board_w - 1);
  //!< neighbors thread which lies on edge should ask appropriately for the bit on the
  //!< other side, these sliding will execute the appropriate shift
  const char bring_right = WIDTH - 1 - __mul24(remaining_cells_w, is_edge_r);
  const char bring_left = WIDTH - 1 - __mul24(remaining_cells_w, is_edge_l);
  //!< junk bits which reside outside the matrix are assigned 0,
  //!< whatever lies inside the matrix 1
  const bboard mask = (bboard)-1 >> __mul24(remaining_cells_w, is_edge_r);

  //!< fetch thread's corresponding bitboard from global mem and send it to
  //!< shared mem
  neighbors[0] = row_c[major_j];
  tiles[tx + 1] = neighbors[0];
  //!< for upper edge and lower edge cases of neighbors block bring also the bitboards
  //!< residing above and below
  major_t = (major_i - 1 + dim_board_h) % dim_board_h;
  row_t = (bboard*)((char*)d_a + major_t * pitch);
  if (tx == 0) {
    tiles[0] = row_t[major_j];
  }
  major_b = (major_i + 1) % dim_board_h;
  row_b = (bboard*)((char*)d_a + major_b * pitch);
  if (tx == bdx - 1) {
    tiles[bdx + 1] = row_b[major_j];
  }
  __syncthreads();

  //!< fetch thread's top and bot neighbor tiles from shared mem
  neighbors[3] = tiles[tx];
  neighbors[6] = tiles[tx + 2];
  __syncthreads();

  //!< fetch thread's left neighbor tile
  const int major_l = (major_j - 1 + dim_board_w) % dim_board_w;
  neighbors[1] = row_c[major_l];
  //!< shift thread to find upper right cell and put left neighbor's last cell
  //!< in first place, send to shared mem
  neighbors[1] = (neighbors[0] << 1) | (neighbors[1] >> bring_left);
  tiles[tx + 1] = neighbors[1];
  //!< block's edge cases
  if (tx == 0) {
    bboard tmp = row_t[major_l];
    tmp = (neighbors[3] << 1) | (tmp >> bring_left);
    tiles[0] = tmp;
  }
  if (tx == bdx - 1) {
    bboard tmp = row_b[major_l];
    tmp = (neighbors[6] << 1) | (tmp >> bring_left);
    tiles[bdx + 1] = tmp;
  }
  __syncthreads();

  //!< fetch thread's top and bot neighbor tiles shifted to 'right'
  neighbors[4] = tiles[tx];
  neighbors[7] = tiles[tx + 2];
  __syncthreads();

  //!< fetch thread's right neighbor tile
  const int major_r = (major_j + 1) % dim_board_w;
  neighbors[2] = row_c[major_r];
  //!< shift thread to find upper right cell and put left neighbor's last cell
  //!< in first place, send to shared mem
  neighbors[2] = (neighbors[0] >> 1) | (neighbors[2] << bring_right);
  tiles[tx + 1] = neighbors[2];
  //!< block's edge cases
  if (tx == 0) {
    bboard tmp = row_t[major_r];
    tmp = (neighbors[3] >> 1) | (tmp << bring_right);
    tiles[0] = tmp;
  }
  if (tx == bdx - 1) {
    bboard tmp = row_b[major_r];
    tmp = (neighbors[6] >> 1) | (tmp << bring_right);
    tiles[bdx + 1] = tmp;
  }
  __syncthreads();

  //!< fetch thread's top and bot neighbor tiles shifted to 'left'
  neighbors[5] = tiles[tx];
  neighbors[8] = tiles[tx + 2];

  if (major_i >= dim_board_h) return;
  if (major_j >= dim_board_w) return;

  // A is the binary repr of the sum of neighbor cells:
  // A0's bit mean 2^0 factor
  // A1's bit mean 2^1 factor
  // A2's bit mean 2^2 factor
  // A3's bit mean 2^3 factor
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
  //!< alive only if (A == 0010 and cell is alive) or (A == 0011)
  row_result[major_j] = ~A[3] & ~A[2] & A[1] & ((~A[0] & neighbors[0]) | A[0]) & mask;
}
