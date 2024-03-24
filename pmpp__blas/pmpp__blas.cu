#include "pmpp__blas.h"

__global__
void pmpp__axpy_kernel(double a, double *x, double *y, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
      y[i] = a*x[i] +y[i];
  }
}

void pmpp__axpy_host(double a, double *x, double *y, int n){
  int i;
  for(i = 0; i < n; i++){
      y[i] = a*x[i] + y[i];
  }
}

__global__
void pmpp__gemm_kernel(double *A, double *B, double *C, const int I, const int J, const int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((col < K) && (row < I)){
        double dot_prod = 0;
        for(int idx = 0; idx < J; idx++){
            dot_prod += A[row*J+idx]*B[idx*K+col];
        }
        C[row*K+col] += dot_prod;
    }
}



__global__
void pmpp__tiled_gemm_kernel(double *A, double *B, double *C, const int I, const int J, const int K, const int TILE_WIDTH){
  // Dinamically allocates the shared memory as a 1D array
  extern __shared__ double shared[];

  // Pointers to the shared arrays sections
  double *A_shared, *B_shared;

  A_shared = shared;
  B_shared = shared + TILE_WIDTH*TILE_WIDTH;

  // Save the threads and blocks idx into registers
  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;

  // Calculates the row and col indexes
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  // Element value accumulator
  double dot_prod = 0.0;

  // Strip Mining outter loop. On each phase, a tile of data is fetched and stored in shared memory
  for(int ph = 0; ph < ceil(J/(double)TILE_WIDTH); ph++){
      // Check if the tile is inside the domain 
      if((row < I) && (ph*TILE_WIDTH + tx) < J){
    	  A_shared[ty*TILE_WIDTH + tx] = A[row*J + ph*TILE_WIDTH + tx];
      }else{
    	  A_shared[ty*TILE_WIDTH + tx] = 0.0;
      }

      if((col < K) && (ph*TILE_WIDTH + ty) < J){
    	  B_shared[ty*TILE_WIDTH + tx] = B[(ph*TILE_WIDTH + ty)*K + col];
      }else{
    	  B_shared[ty*TILE_WIDTH + tx] = 0.0;
      }  

      // Wait for all threads in the block to complete the data loading
      __syncthreads();

      // Performs the dot product with the data loaded on this phase
      for(int idx = 0; idx < TILE_WIDTH; idx++){
          dot_prod += A_shared[ty*TILE_WIDTH + idx]*B_shared[idx*TILE_WIDTH + tx];
      } 

      // Wait for all threads in the block to complete the calculation
      __syncthreads();   
  }

  // Saves the dot product to C[row][col] position
  if((row < I) && (col < K)){
      C[row*K + col] += dot_prod;
  }
}

void pmpp__gemm_host(double *A, double *B, double *C, const int I,const int J,const int K){
  for(int i = 0; i < I; i++){        
    for(int k = 0; k < K; k++){
      double dot_prod = 0;
        for(int idx = 0; idx < J; idx++){
            dot_prod += A[i*J+idx]*B[idx*K+k];
        }   
        C[i*K+k] += dot_prod;     
      }          
  }    
}
