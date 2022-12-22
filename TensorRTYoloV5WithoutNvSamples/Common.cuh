#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "NvInfer.h"

#include <stdio.h>



#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err), cudaGetErrorString(err), file, line);
		exit(int(err));
	}
}

__global__ void select_max_kernel(float* arr, int* index, const int len);