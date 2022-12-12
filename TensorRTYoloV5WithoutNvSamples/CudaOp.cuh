#include "cuda_runtime.h"
#include "device_launch_parameters.h"



cudaError_t addWithCuda2(int* c, const int* a, const int* b, unsigned int size);

cudaError_t find_the_max_class_score(float* cuda_output, float* class_index, unsigned int size);