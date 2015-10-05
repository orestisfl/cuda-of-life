#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "../utils/utils.h"

#define NTHREADS 4

int count_neighbors(int x0, int x1, int x2, int y0, int y1, int y2);

/* table is where we store the actual data,
 * help_table is used for the calculation of a new generation */
int* table;
int* help_table;
int N;

inline int count_neighbors(int up, int owni, int down, int left, int ownj, int right) {
    return
        table[POS(up   , left)] +
        table[POS(up   , ownj)] +
        table[POS(up   , right)] +
        table[POS(owni , left)] +
        table[POS(owni , right)] +
        table[POS(down, left)] +
        table[POS(down, ownj)] +
        table[POS(down, right)] ;
}

int* prev_of;
int* next_of;

void serial_compute(void) {
    int i, j, left, right, up, down;
    unsigned int alive_neighbors;

    for (i = 0; i < N; ++i) {
        up = (i - 1 + N) % N;
        down = (i + 1) % N;

        left = (j - 1 + N) % N;
        right = (j + 1) % N;
        alive_neighbors = count_neighbors(up, i, down, left, j, right);
        help_table[POS(i, j)] = (alive_neighbors == 3) || (alive_neighbors == 2 &&
                                                           table[POS(i, j)]) ? 1 : 0;
    }
}

swap(&table, &help_table);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s FILE dimension\n", argv[0]);
        exit(1);
    }

    int N_RUNS = 10;
    char* filename = argv[1];
    N = atoi(argv[2]);
    int total_size = N * N;

    if (argc == 4)
        N_RUNS = atoi(argv[3]);

    printf("Reading %dx%d table from file %s\n", N, N, filename);
    table = (int*) malloc(total_size * sizeof(int));
    help_table = (int*) malloc(total_size * sizeof(int));
    read_from_file(table, filename, N, N);
    printf("Finished reading table\n");
    struct timeval startwtime, endwtime;
    gettimeofday(&startwtime, NULL);

    for (int i = 0; i < N_RUNS; ++i) {
        serial_compute();
        print_table(table, N, N);
    }

    gettimeofday(&endwtime, NULL);
    double time = (double)((endwtime.tv_usec - startwtime.tv_usec)
                           / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf(ANSI_COLOR_RED"OMP"ANSI_COLOR_RESET
           " time to run: "ANSI_COLOR_RED"%f"ANSI_COLOR_RESET" ms\n", time * 1000);
    save_table(table, N, N, "omp-results.bin");
    free(table);
    free(help_table);
}
