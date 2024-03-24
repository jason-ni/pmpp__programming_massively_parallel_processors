#ifndef PMPP__BLAS_H_
#define PMPP__BLAS_H_

/**
* Kernel that performs the axpy BLAS operation
* @param a The constant "a"
* @param x The array "x"
* @param y The array "y"
* @param n The lenght of the arrays
*/
__global__ void pmpp__axpy_kernel(double a, double *x, double *y, int n);

/**
* The host function that performs the axpy BLAS operation
* @param a The constant "a"
* @param x The array "x"
* @param y The array "y"
* @param n The lenght of the arrays
*/
void pmpp__axpy_host(double a, double *x, double *y, int n);


/**
* Kernel that performs a naive (without tiling) GEMM operation
* @param A The matrix A
* @param B The matrix B
* @param C The matrix C
* @param I The number of rows of the matrix A
* @param J The number of columns of the matrix A (The number of rows of the matrix B)
* @param K The number of columns of the matrix B
*/
__global__ void pmpp__gemm_kernel(double *A, double *B, double * C, const int I, const int J, const int K);

/**
* Kernel that performs a tiled GEMM operation
* @param A The matrix A
* @param B The matrix B
* @param C The matrix C
* @param I The number of rows of the matrix A
* @param J The number of columns of the matrix A (The number of rows of the matrix B)
* @param K The number of columns of the matrix B
* @param TILE_WIDTH The tile width for square tiles
*/
__global__ void pmpp__tiled_gemm_kernel(double *A, double *B, double *C, const int I, const int J, const int K, const int TILE_WIDTH);

/**
* Host function that performs a naive GEMM operation
* @param A The matrix A
* @param B The matrix B
* @param C The matrix C
* @param I The number of rows of the matrix A
* @param J The number of columns of the matrix A (The number of rows of the matrix B)
* @param K The number of columns of the matrix B
*/
void pmpp__gemm_host(double *A, double *B, double *C, const int I,const int J,const int K);

#endif /* PMPP__BLAS_H_ */
