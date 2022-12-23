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







//cuda nms
#include <math.h>

int const threadsPerBlock = sizeof(unsigned long long int) * 8; // threadsPerBlock 定义在这里，等于unsigned longlong的位数

__device__ inline bool devIoU(float const* const a, float const* const b, const int offset, const float threshold)
{
	// 计算两个bbox的iou，__device__说明在cuda上执行，将被nms_cuda调用
	float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
	float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
	float width = fmaxf(right - left + offset, 0.f),
		height = fmaxf(bottom - top + offset, 0.f);
	float interS = width * height;
	float Sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
	float Sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
	return interS > threshold * (Sa + Sb - interS);
}

__global__ void nms_cuda(const int n_boxes, const float iou_threshold, const int offset, const float* dev_boxes, unsigned long long* dev_mask)
{
	const int row_start = blockIdx.y;
	const int col_start = blockIdx.x;
	const int tid = threadIdx.x;

	if (row_start > col_start) return; // 只计算上三角的iou矩阵

	const int row_size = fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
	// 最大为threadsPerBlock，因为n_boxes可能不能被threadsPerBlock整除，获得余数
	const int col_size = fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

	__shared__ float block_boxes[threadsPerBlock * 4];
	// 共享内存，把同一线程块中频繁访问的64个bbox的信息放到共享内存
	// 共享内存对同一线程块中的所有内存共享
	// 这里每个线程，负责把一个bbox放到共享内存中 
	if (tid < col_size) {
		block_boxes[tid * 4 + 0] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 0];
		// dev_boxes是一维数组的首地址
		block_boxes[tid * 4 + 1] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 1];
		block_boxes[tid * 4 + 2] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 2];
		block_boxes[tid * 4 + 3] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 3];
	}
	__syncthreads(); // 同步！使用共享内存一定要同步，等64个线程把bbox放到共享内存后，再计算后面的iou
	//波浪线 VS不识别，但是cuda能识别，可以无视

	if (tid < row_size) {
		const int cur_box_idx = threadsPerBlock * row_start + tid;
		const float* cur_box = dev_boxes + cur_box_idx * 4;
		int i = 0;
		unsigned long long int t = 0;
		int start = 0;
		if (row_start == col_start) {
			start = tid + 1; // 对角线上的跳过，不算iou
		}
		for (i = start; i < col_size; i++) {
			// 每个线程要算col_size次iou
			if (devIoU(cur_box, block_boxes + i * 4, offset, iou_threshold)) {
				t |= 1ULL << i; // 如果iou大于阈值，通过位运算，t为64位0 or 1，把t的第i位设为1
			}
		}
		dev_mask[cur_box_idx * gridDim.y + col_start] = t; // 修改mask[cur_box_idx][col_start]为t
	}
}
