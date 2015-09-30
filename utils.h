#ifndef UTILS_H
#define UTILS_H
#include <stdint.h>

typedef uint64_t bboard;
typedef unsigned int uint;
#define ONE 1UL
#define WIDTH 8
#define DFL_RUNS 10
#define THRESHOLD 0.4

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define CEIL_DIV(X, Y) (1 + (((X) - 1) / (Y)))

extern void read_from_file(int* X, const char* filename, int dim);
extern void save_table(int* X, int dim, const char* filename);
extern void generate_table(int* X, int M, int N);
extern void print_table(int* A, int M, int N);

#endif
