#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "NvInferRuntime.h"
#include <cuda_runtime_api.h>

#include "common.cuh"


cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

int find_all_max_class_score(float* cuda_output, int outputBoxecount, float** host_objects_p);