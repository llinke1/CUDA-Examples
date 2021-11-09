/***********************************************************************************/
/* Example for using CUDA to calculate the Paircount needed for Landy Szalay *******/
/* Barebone example, does not include proper error handling for better readibility */
/***********************************************************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono> // For time measurement
#include <random> // For random numbers
#include <stdio.h>
#define MAX_THREADS_PER_BLOCK 1024 // maximal threads within one block

#define THETAMIN 0.1 // Minimal theta for binning [arcmin]
#define THETAMAX 5 // Maximal theta for binning [arcmin]
#define NBINS 512 // Number of bins


/*
Calculates the Paircount of two 2-D galaxy catalogs on GPU
Kernel Function: Is evaluated by each thread
@param x1 x-positions of first catalog [arcmin]
@param y1 y-positions of first catalog [arcmin]
@param Ngal1 number of galaxies in first catalog
@param x2 x-positions of second catalog [arcmin]
@param y2 y-positions of second catalog [arcmin]
@param Ngal2 number of galaxies in second catalog
@param paircount Array which will contain binned paircount (binned logarithmically between THETAMIN and THETAMAX)
*/
__global__ void calculatePaircountGPU(double* x1, double* y1, int Ngal1, double* x2, double* y2, int Ngal2,
			                                int* paircount)
{
  double theta_binwidth=log(THETAMAX/THETAMIN)/NBINS; //Logarithmic bin width

  for (int i = threadIdx.x; i < Ngal1; i += blockDim.x * gridDim.x)
  {
    double x1_ = x1[i];
    double y1_ = y1[i];
      
    for (int j = 0; j < Ngal2; j++)
    {
	    // Get Distance between galaxies
	    double dx = x1_ - x2[j];
	    double dy = y1_ - y2[j];
	  
      double r = sqrt(dx * dx + dy * dy); // 2D separation between galaxies on the sky
	  
	    // Get index in paircount corresponding to r
	    int ix; // Bin index
	    if (r > THETAMIN && r <= THETAMAX) // Check if r within bins
      {
	      ix = floor(log(r / THETAMIN) / theta_binwidth);
	      
	      // Add 1 to relevant bin
	      atomicAdd(&paircount[ix], 1); // AtomicAdd Makes sure that no two threads write simultaneously to the same location!
      };
    }
  }
}

/*
Calculates the Paircount of two 2-D galaxy catalogs on CPU
@param x1 x-positions of first catalog [arcmin]
@param y1 y-positions of first catalog [arcmin]
@param Ngal1 number of galaxies in first catalog
@param x2 x-positions of second catalog [arcmin]
@param y2 y-positions of second catalog [arcmin]
@param Ngal2 number of galaxies in second catalog
@param paircount Array which will contain binned paircount (binned logarithmically between THETAMIN and THETAMAX)
*/
void calculatePaircountCPU(double* x1, double* y1, int Ngal1, double* x2, double* y2, int Ngal2,
			                      int* paircount)
{
  double theta_binwidth=log(THETAMAX/THETAMIN)/NBINS; //Logarithmic bin width

  for (int i = 0; i < Ngal1; i ++)
  {
    double x1_ = x1[i];
    double y1_ = y1[i];
      
    for (int j = 0; j < Ngal2; j++)
    {
	   // Get Distance between galaxies
	  
	    double dx = x1_ - x2[j];
	    double dy = y1_ - y2[j];
	    double r = sqrt(dx * dx + dy * dy); // 2D separation between galaxies on the sky
	  
	    // Get index in paircount corresponding to r
	    int ix; // Bin index
	    if (r > THETAMIN && r <= THETAMAX) // Check if r within bins
      {
	      ix = floor(log(r / THETAMIN) / theta_binwidth);  
	      paircount[ix]+=1; // Add 1 to relevant bin
      };
    };
  };
}


int main()
{
    // Declare and define x and y arrays on CPU (Should actually be read from file)
    const int arraySize = 10240; // Number of elements in array
    static double x1[arraySize]; //Summand 1
    static double x2[arraySize]; // Summand 2
    static double y1[arraySize]; // Result array
    static double y2[arraySize]; // Result array

    static int paircount[NBINS];
    
    // Fill arrays with random numbers between 0 and theta_max
    // All have the same seed, but for this example this should not matter
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, THETAMAX);
    for (int i = 0; i < arraySize; i++)
    {
        x1[i] = dist(mt);
	      x2[i] = dist(mt);
        y1[i] = dist(mt);
        y2[i]=dist(mt);
    };
    
    // GPU CALCULATIONS

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate GPU buffers for the three arrays
    double* dev_x1, * dev_x2, * dev_y1, * dev_y2;
    int * dev_paircount;
    cudaMalloc((void**)&dev_x1, arraySize * sizeof(double));   
    cudaMalloc((void**)&dev_x2, arraySize * sizeof(double)); 
    cudaMalloc((void**)&dev_y1, arraySize * sizeof(double)); 
    cudaMalloc((void**)&dev_y2, arraySize * sizeof(double));  
    cudaMalloc((void**)&dev_paircount, NBINS*sizeof(int));
   
    // Copy data to device
    cudaMemcpy(dev_x1, x1, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x2, x2, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y1, y1, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y2, y2, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // Do calculation
    int blocks = arraySize/MAX_THREADS_PER_BLOCK; //Number of Thread blocks
    int threads = MAX_THREADS_PER_BLOCK; // Number of threads per block
    calculatePaircountGPU << < blocks, threads >> > (dev_x1, dev_y1, arraySize, dev_x2, dev_y2, arraySize, dev_paircount); //Launch GPU Kernel, 1 thread per Galaxy from catalog 1

    // Copy result from device to host
    cudaMemcpy(paircount, dev_paircount, NBINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Clean up: Free space on device
    cudaFree(dev_x1);
    cudaFree(dev_x2);
    cudaFree(dev_y1);
    cudaFree(dev_y2);
    cudaFree(dev_paircount);

    // Stop time measurement
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gpu = stop - start;
    printf("Needed %f seconds for calculating paircount of %d galaxies on GPU \n", elapsed_gpu.count(), arraySize);


    // CPU CALCULATIONS
    // Start time measurement
    start = std::chrono::high_resolution_clock::now();
    
    calculatePaircountCPU(x1, y1, arraySize, x2, y2, arraySize, paircount);
    
    // Stop time measurement
    stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = stop - start;
    printf("Needed %f seconds for calculating paircount of %d galaxies on CPU \n", elapsed_cpu.count(), arraySize);

    return 0;
}

