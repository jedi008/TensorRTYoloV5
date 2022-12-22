#include "Common.cuh"

__global__ void select_max_kernel(float* arr, int* index_r, const int len)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int n = bid * blockDim.x + tid;
	extern __shared__ int index[];
	index[tid] = (n < len) ? tid : 0;
	__syncthreads();
	
	for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
	{
		if (tid < offset)
		{
			if (arr[index[tid]] < arr[index[tid + offset]])
			{
				index[tid] = tid + offset;
			}
			__syncthreads();
		}
	}

	for (int offset = 16; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			if (arr[index[tid]] < arr[index[tid + offset]])
			{
				index[tid] = index[tid + offset];
			}
			__syncwarp();
		}
	}

	if (tid == 0)
	{
		index_r[0] = index[0];
	}
}
