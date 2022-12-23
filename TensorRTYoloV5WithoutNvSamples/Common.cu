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

__global__ void rankSort(int* d_a, int* d_b)
{
	int tid = threadIdx.x;
	int ttid = threadIdx.x + blockIdx.x * blockDim.x;
	int val = d_a[ttid];
	__shared__ int cache[5];
	cache[tid] = d_a[ttid];
	__syncthreads();
	//统计每个数字比它小的数字有几个
	int count = 0;
	for (int j = 0; j < 5; j++)
	{
		if (val > cache[j])
		{
			count++;
		}
	}
	__syncthreads();
	d_b[count] = val;
}
