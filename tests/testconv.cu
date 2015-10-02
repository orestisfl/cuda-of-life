#define TESTING
#include "../main.cu"

// fill array A with zeros
__global__
static void zero_k(int* A, size_t dim) {
    for (int i = 0; i < dim * dim; i ++) {
        A[i] = 0;
    }
}


int main(void) {
    const int dim = 1000;
    const size_t mem_size = dim * dim * sizeof(int);
    const int dim_board_w = CEIL_DIV(dim, WIDTH);
    const int dim_board_h = CEIL_DIV(dim, HEIGHT);
    cudaFree(0); // init device
    dim3 block(8, 8);
    dim3 grid(32, 16);

    int* table;
    table = (int*) malloc(mem_size);
    read_from_file(table, "test_in.bin", dim);
    //    print_table(table, dim);

    int* d_table;
    cudaMalloc((void**) &d_table,  mem_size);
    cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);

    bboard* d_board;
    size_t pitch;
    cudaMallocPitch((void**)&d_board, &pitch, dim_board_w * sizeof(bboard), dim_board_h);
    cudaCheckErrors("device pitch allocation of GOL matrix failed", __FILE__, __LINE__);

    cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);
    free(table);

    convert_to_tiled <<< grid, block >>> (d_table, d_board, dim, pitch);
    cudaCheckErrors("convert_to_tiled failed", __FILE__, __LINE__);
    
    zero_k <<< 1, 1 >>> (d_table, dim);
    
    convert_from_tiled <<< grid, block >>> (d_table, d_board, dim, pitch);
    cudaCheckErrors("convert_from_tiled failed", __FILE__, __LINE__);

    table = (int*) malloc(mem_size);
    cudaMemcpy(table, d_table, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy from device to host memory failed", __FILE__, __LINE__);

    //    print_table(table, dim);
    save_table(table, dim, "test_out.bin");
}
