#include <stdio.h>
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"
#include "../../utils.h"

// load a matrix from file
void load_matrix(float *matrix, int rows, int cols, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%f", &matrix[i*cols+j])!= 1) {
                fprintf(stderr, "Error: could not read matrix element (%d, %d) from file %s\n", i, j, filename);
                exit(1);
            }
        }
    }
    fclose(file);
}

// print out matrix
void print_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i*cols+j]);
        }
        printf("\n");
    }
}

__global__ void matrix_mul_kernel(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    int cRow = blockIdx.y * blockDim.y + threadIdx.y;
    int cCol = blockIdx.x * blockDim.x + threadIdx.x;
    if (cRow < rowsA && cCol < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += A[cRow*colsA+k] * B[k*colsB+cCol];
        }
        C[cRow*colsB+cCol] = sum;
    }
}

void matrix_mul_gpu(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    // define device memory
    float *A_d, *B_d, *C_d;

    HOST_TIC(0);
    // allocate device memory
    CCE(cudaMalloc(&A_d, rowsA*colsA*sizeof(float)));
    CCE(cudaMalloc(&B_d, colsA*colsB*sizeof(float)));
    CCE(cudaMalloc(&C_d, rowsA*colsB*sizeof(float)));

    // copy host memory to device memory
    CCE(cudaMemcpy(A_d, A, rowsA*colsA*sizeof(float), cudaMemcpyHostToDevice));
    CCE(cudaMemcpy(B_d, B, colsA*colsB*sizeof(float), cudaMemcpyHostToDevice));
    HOST_TOC(0);

    // define grid and block dimensions
    int block_size = 16;
    dim3 numThreadsPerBlock(block_size, block_size);
    dim3 numBlocks((colsB + numThreadsPerBlock.x - 1)/block_size, (rowsA + numThreadsPerBlock.y - 1)/block_size);

    // call kernel
    HOST_TIC(1);
    matrix_mul_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, rowsA, colsA, colsB);
    HOST_TOC(1);

    // copy device memory back to host memory
    CCE(cudaMemcpy(C, C_d, rowsA*colsB*sizeof(float), cudaMemcpyDeviceToHost));

    // free device memory
    CCE(cudaFree(A_d));
    CCE(cudaFree(B_d));
    CCE(cudaFree(C_d));
}

int main() {
    float *A, *B, *C;
    int rowsA, colsA, colsB;
    rowsA = 4;
    colsA = 4;
    colsB = 10;
    const char *filenameA = "A.txt";
    const char *filenameB = "B.txt";

    A = (float*)malloc(rowsA*colsA*sizeof(float));
    B = (float*)malloc(colsA*colsB*sizeof(float));
    C = (float*)malloc(rowsA*colsB*sizeof(float));
    load_matrix(A, rowsA, colsA, filenameA);
    load_matrix(B, colsA, colsB, filenameB);
    print_matrix(A, rowsA, colsA);
    print_matrix(B, colsA, colsB);

    matrix_mul_gpu(A, B, C, rowsA, colsA, colsB);
    print_matrix(C, rowsA, colsB);
}

