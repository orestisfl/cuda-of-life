#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Position of i-th row j-th element using our current data arrangement. */

void read_from_file(int* X, const char* filename, int dim) {
    FILE* fp = fopen(filename, "r+");
    int size = fread(X, sizeof(int), dim * dim, fp);
    printf("elements: %d\n", size);
    fclose(fp);
}

void save_table(int* X, int dim, const char* filename) {
    FILE* fp;
    printf("Saving table in file %s\n", filename);
    fp = fopen(filename, "w+");
    fwrite(X, sizeof(int), dim * dim, fp);
    fclose(fp);
}

void generate_table(int* X, int dim) {
    srand(time(NULL));
    int counter = 0;

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            X[i * dim + j] = ((float)rand() / (float)RAND_MAX) < THRESHOLD;
            counter += X[i * dim + j];
        }
    }

    printf("Number of non zero elements: %d\n", counter);
    printf("Percent: %f\n", (float)counter / (float)(dim * dim));
}

void print_table(int* A, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            fprintf(stderr, "%s%d "ANSI_COLOR_RESET, A[i * dim + j] ? ANSI_COLOR_BLUE : ANSI_COLOR_RED,
                    A[i * dim + j]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}
