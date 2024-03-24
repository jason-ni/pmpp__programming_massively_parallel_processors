#ifndef NVIXNU__ERROR_UTILS_H_
#define NVIXNU__ERROR_UTILS_H_

/**
* Macro that receives the cudaError_t error and call the inline function check_cuda_error passing the error, the file and the line which the error occurs
*/
#define CCE(error) { check_cuda_error(error, __FILE__, __LINE__); }

/**
* Macro that gets the last CUDA error and call the inline function check_cuda_error passing the error, the file and the line which the error occurs.
* This macro is useful to capture kernel errors
*/
#define CCLE() { cudaError_t error = cudaGetLastError(); check_cuda_error(error, __FILE__, __LINE__); }

/**
* Function called by the macros CCE and CCLE to show the error messages and finish the execution
* @param error The error returned by CUDA API functions
* @param file Name of the file where the error occurs
* @param line The line where the error occurs
*/
static inline void check_cuda_error(cudaError_t error, const char * file, int line){
   if (error != cudaSuccess){
      printf("\nCUDA ERROR AT %s line %d: %s", file, line, cudaGetErrorString(error));
      exit(error);
   }
}

#endif /* NVIXNU__ERROR_UTILS_H_ */
