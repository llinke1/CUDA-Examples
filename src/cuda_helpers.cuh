#ifndef CUDA_HELPERS
#define CUDA_HELPERS
#include <iostream>
#define CUDA_SAFE_CALL(ans) { cuda_safe_call((ans), __FILE__, __LINE__); }
inline void cuda_safe_call(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA error at %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#endif