/***********************************************************************************/
/* Example for using CUDA to multiply two large matrices **********************************/
/* Barebone example, does not include proper error handling for better readibility */
/***********************************************************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono> // For time measurement

#include <stdio.h>
#define MAX_THREADS_PER_BLOCK 1024 // maximal threads within one block


/*
multiplies two matrices
Kernel Function: Is evaluated by each thread
@param c Result array
@param a Summand 1
@param b Summand 2
@param n Number of rows
@param m Number of columns
*/
__global__ void addKernel(int* c, const int* a, const int* b, int n, int m) //__global__ means: Is executed on GPU and can be called from CPU
{
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Index of this thread
    if (i < n)
    {
        for (int j = 0; j < m; j++)
        {
            c[i]
        }
    }
    c[i] = a[i] + b[i]; // Do summation
}

int main()
{
    // Declare and define arrays on CPU
    const int arraySize = 2048; // Number of elements in array
    int a[arraySize]; //Summand 1
    int b[arraySize]; // Summand 2
    int c[arraySize]; // Result array

    // Fill arrays
    for (int i = 0; i < arraySize; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    };
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
    int blocks = arraySize / MAX_THREADS_PER_BLOCK; //Number of Thread blocks
    int threads = MAX_THREADS_PER_BLOCK; // Number of threads per block
    addKernel << < blocks, threads >> > (dev_c, dev_a, dev_b); //Launch GPU Kernel, 1 thread per element

    // Copy result from device to host
    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Stop time measurement
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

    // Output
    printf("Result is \n");
    for (int i = 0; i < arraySize; i++)
    {
        printf("%d \n", c[i]);
    };

    printf("Needed %f seconds for adding on GPU \n", elapsed.count());

    // Clean up: Free space on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}

