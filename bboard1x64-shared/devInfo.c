#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp props;
    int devCount;
    cudaGetDeviceCount(&devCount);

    if (devCount > 0) {
        cudaGetDeviceProperties(&props, 0);
        printf("%d%d", props.major, props.minor);
    } else return 1;
}
