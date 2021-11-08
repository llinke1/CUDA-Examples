/***********************************************************************************/
/* Example for using CUDA to multiply two large matrices  **********************************/
/* Barebone example, does not include proper error handling for better readibility */
/***********************************************************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono> // For time measurement

#include <iostream>
#define MAX_THREADS_PER_BLOCK 1024 // maximal threads within one block


/*
Multiplys two equal-size square-matrices on GPU
Kernel Function: Is evaluated by each thread
__global__ means: Is executed on GPU and can be called from CPU
@param c Result matrix
@param a Matrix 1
@param b Matrix 2
@param N number of rows and cols of a, b, and c
*/
__global__ void multiplyGPU(int *c, const int *a, const int* b, const int N)
{
  //Row and Col of Result matrix calculated in this thread
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col=threadIdx.x + blockIdx.x * blockDim.x;

  int tmp=0; // Temporary result storage to avoid calling c too often
    
  if(row<N && col<N)
  {
	  for(int i=0; i<N; i++)
	  {
	    tmp += a[row*N+i]*b[i*N+col]; //C_ij=sum_k A_ik * B_kj
	  };
    c[row*N+col]=tmp;
  };
}

/*
Multiplys two equal-size square-matrices on CPU
@param c Result matrix
@param a Matrix 1
@param b Matrix 2
@param N number of rows and cols of a, b, and c
*/
void multiplyCPU(int *c, const int* a, const int* b, int N)
{
  
  for(int i=0; i<N; i++) // Loop through all rows of result matrix
  {
    for(int j=0; j<N; j++) // Loop through all cols of result matrix
	  {
	    for(int k=0; k<N; k++) // Loop through sum
	    {
	      c[i*N+j]+=a[i*N+k]*b[k*N+j]; // C_ij=sum_k A_ik*B_kj
	    };
	  };
  };
}

int main()
{
  // Declare and define arrays on CPU
  const int N=1024; // Number of rows/cols of matrix
  const int arraySize = N*N; // Number of elements per matrix
  static int a[arraySize]; // Matrix 1
  static int b[arraySize]; // Matrix 2
  static int c[arraySize]; // Result Matrix

  // Fill Matrices
  for (int i = 0; i < arraySize; i++)
  {
    a[i] = i;
	  b[i] = 2 * i;
  };
    
  // GPU CALCULATIONS
  
  // Start time measurement
  auto start = std::chrono::high_resolution_clock::now();
    
  // Allocate GPU buffers for the three arrays
  int* dev_a, * dev_b, * dev_c;
  cudaMalloc((void**)&dev_a, arraySize * sizeof(int)); // Allocate memory for summand 1 array
  cudaMalloc((void**)&dev_b, arraySize * sizeof(int)); // Allocate memory for summand 2 array
  cudaMalloc((void**)&dev_c, arraySize * sizeof(int)); // Allocate memory for result array


  // Copy Summands to device
  cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

  // Do calculation
  dim3 blocks(1,1); //Trick: 2-Dimensional Blocks!
  dim3 threads(N,N); // Trick: N threads along dimension 1 of a block, and N threads along dimension 2 of a block
  multiplyGPU <<< blocks, threads >>> (dev_c, dev_a, dev_b, N); //Launch GPU Kernel, 1 thread per element of result matrix

  // Copy result from device to host
  cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

  // Clean up: Free space on device
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  

  // Stop time measurement
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_gpu = stop - start;
  printf("Needed %f seconds for multiplying two %dx%d matrices on GPU \n", elapsed_gpu.count(), N, N);

  // CPU CALCULATIONS
  // Start time measurement
  start = std::chrono::high_resolution_clock::now();
    
  multiplyCPU(c, a, b, N);

    
  // Stop time measurement
  stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_cpu = stop - start;
     
  
  printf("Needed %f seconds for multiplying two %dx%d matrices on CPU \n", elapsed_cpu.count(), N, N);
  return 0;
}

