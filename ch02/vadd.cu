#include "../utils.h"

__host__ __device__ float f(float a, float b) {
    return a + b;
}

void vecadd_cpu(float *x, float *y, float *z, unsigned int N) {
    for (unsigned int i = 0; i < N; i++) {
        z[i] = f(x[i], y[i]);
    }
}

__global__ void vecadd_kernel(float *x, float *y, float *z, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        z[i] = f(x[i], y[i]);
    }
}

void vecadd_gpu(float *x, float *y, float *z, unsigned int N) {
    // Allocate gpu memory
    float *x_d, *y_d, *z_d;
    cudaMalloc(&x_d, N*sizeof(float));
    cudaMalloc(&y_d, N*sizeof(float));
    cudaMalloc(&z_d, N*sizeof(float));

    // Copy data to gpu
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    const unsigned int threads_per_block = 512;
    const unsigned int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    vecadd_kernel<<<blocks_per_grid, threads_per_block>>>(x_d, y_d, z_d, N);

    

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main(int argc, char** argv) {
    cudaDeviceSynchronize();

    unsigned int N = (argc > 1)?(atoi(argv[1])):(1<<25);
    float* x = (float*)malloc(N*sizeof(float));
    float* y = (float*)malloc(N*sizeof(float));
    float* z = (float*)malloc(N*sizeof(float));

    for (unsigned int i = 0; i < N; i++) {
        x[i] = rand();
        y[i] = rand();
    }

    HOST_TIC(0);
    vecadd_cpu(x, y, z, N);
    HOST_TOC(0);

    HOST_TIC(1);
    vecadd_gpu(x, y, z, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    HOST_TOC(1);
    free(x);
    free(y);
    free(z);
}