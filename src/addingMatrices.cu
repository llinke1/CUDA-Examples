/***********************************************************************************/
/* Example for using CUDA to add two large arrays (a+b=c) ***************************/
/* Barebone example, does not include proper error handling for better readibility */
/***********************************************************************************/


#include "cuda_runtime.h" // To use CUDA
#include "device_launch_parameters.h" // Some CUDA "stuff"
#include <chrono> // For time measurement

#include <iostream> // In/Output

#define MAX_THREADS_PER_BLOCK 1024 // maximal threads within one block (Might vary with GPU specs)


/*
Adds two (equal length) arrays on GPU
Kernel Function: Is evaluated by each thread
__global__ means: Is executed on GPU and can be called from CPU
@param c Result array
@param a Summand 1
@param b Summand 2
@param N number of elements in a, b, and c
*/
__global__ void addGPU(int *c, const int *a, const int* b, const int N)
{  
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Index of this thread
    if(i<N) // Make sure we are not outside of the arrays
      {
	    c[i] = a[i] + b[i]; // Do summation
      }
}

/*
Adds two (equal lenght) arrays on CPU
@param c Result array
@param a Summand 1
@param b Summand 2
@param N number of elements in a, b, and c
*/
void addCPU(int *c, const int* a, const int* b, int N)
{
  for(int i=0; i<N; i++) // Go through whole arrays
    {
	  c[i]=a[i] + b[i];
    };
}

int main()
{
    // Declare and define arrays on CPU
    const int arraySize = 1024*1024; // Number of elements in array
    static int a[arraySize]; //Summand 1
    static int b[arraySize]; // Summand 2
    static int c[arraySize]; // Result array

    // Fill arrays
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
    int blocks = arraySize/MAX_THREADS_PER_BLOCK; //Number of Thread blocks
    int threads = MAX_THREADS_PER_BLOCK; // Number of threads per block
    addGPU <<< blocks, threads >>> (dev_c, dev_a, dev_b, arraySize); //Launch GPU Kernel, 1 thread per element

    // Copy result from device to host
    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up: Free space on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Stop time measurement
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gpu = stop - start;
    
    printf("Needed %f seconds for adding %d elements on GPU \n", elapsed_gpu.count(), arraySize);

    // CPU CALCULATIONS

    // Start time measurement
    start = std::chrono::high_resolution_clock::now();
    
    addCPU(c, a, b, arraySize);
    
    // Stop time measurement
    stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = stop - start;
   
 
    printf("Needed %f seconds for adding %d elements on CPU \n", elapsed_cpu.count(), arraySize);

    return 0;
}

