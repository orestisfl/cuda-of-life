#ifndef UTILS_H
#define UTILS_H
#include <stdint.h>
#include <cuda_runtime.h>

typedef uint32_t bboard;
typedef uint64_t ext_bboard;
typedef unsigned int uint;
#define ONE 1UL
#define ZERO 0UL
#define WIDTH 8
#define HEIGHT 4
#define EXT_WIDTH (WIDTH + 2)
#define EXT_HEIGHT (HEIGHT + 2)
#define DFL_RUNS 10

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define CEIL_DIV(X, Y) (1 + (((X) - 1) / (Y)))

#define POS(i,j) ((i) * WIDTH + (j))

#define SET_BOARD(val, i, j) val |= (ONE << POS(i, j))
#define CLEAR_BOARD(val, i, j) val &= ~(ONE << POS(i, j))
#define TOGGLE_BOARD(val, i, j) val ^= (ONE << POS(i, j))
#define BOARD_IS_SET(val, i, j) ((bool)((val) & (ONE << POS(i, j))))

#define EXT_POS(i,j) (i * EXT_WIDTH + j)
#define EXT_SET_BOARD(val, i, j) val |= (ONE << EXT_POS(i, j))
#define EXT_BOARD_IS_SET(val, i, j) ((bool)((val) & (ONE << EXT_POS(i, j))))

#define BBOARD_LEFT_COL_MASK 16843009u
#define BBOARD_RIGHT_COL_MASK 2155905152u
#define BBOARD_UPPER_ROW_MASK 255u
#define BBOARD_BOTTOM_ROW_MASK 4278190080u
#define BBOARD_CENTER_MASK 8289792u

#define EXT_BBOARD_CENTER_MASK 561299073792000lu

#if (__CUDA_ARCH__ >= 200)
    #define CUDA_PRINT(...) printf(__VA_ARGS__)
#else
    #define CUDA_PRINT(...)
#endif

extern void read_from_file(int* X, const char* filename, int dim);
extern void save_table(int* X, int dim, const char* filename);
extern void generate_table(int* X, int dim);
extern void print_table(int* A, int dim);

extern __global__ void convert_to_tiled(const int* d_table, bboard* d_a, const size_t dim,
                                        const size_t pitch);
extern __global__ void convert_from_tiled(int* d_table, const bboard* d_a, const int dim,
                                          const size_t pitch);
extern __global__ void calculate_next_generation(const bboard* d_a,
                                                 bboard* d_result,
                                                 const int dim,
                                                 const int dim_board_w,
                                                 const int dim_board_h,
                                                 const size_t pitch,
                                                 const int remaining_cells_w,
                                                 const int remaining_cells_h
                                                );
extern __global__ void calculate_next_generation_no_rem(const bboard* d_a,
                                                        bboard* d_result,
                                                        const int dim,
                                                        const int dim_board_w,
                                                        const int dim_board_h,
                                                        const size_t pitch
                                                       );
#endif
