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
	//ͳ��ÿ�����ֱ���С�������м���
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

int const threadsPerBlock = sizeof(unsigned long long int) * 8; // threadsPerBlock �������������unsigned longlong��λ��

__device__ inline bool devIoU(float const* const a, float const* const b, const int offset, const float threshold)
{
	// ��������bbox��iou��__device__˵����cuda��ִ�У�����nms_cuda����
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

	if (row_start > col_start) return; // ֻ���������ǵ�iou����

	const int row_size = fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
	// ���ΪthreadsPerBlock����Ϊn_boxes���ܲ��ܱ�threadsPerBlock�������������
	const int col_size = fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

	__shared__ float block_boxes[threadsPerBlock * 4];
	// �����ڴ棬��ͬһ�߳̿���Ƶ�����ʵ�64��bbox����Ϣ�ŵ������ڴ�
	// �����ڴ��ͬһ�߳̿��е������ڴ湲��
	// ����ÿ���̣߳������һ��bbox�ŵ������ڴ��� 
	if (tid < col_size) {
		block_boxes[tid * 4 + 0] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 0];
		// dev_boxes��һά������׵�ַ
		block_boxes[tid * 4 + 1] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 1];
		block_boxes[tid * 4 + 2] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 2];
		block_boxes[tid * 4 + 3] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 3];
	}
	__syncthreads(); // ͬ����ʹ�ù����ڴ�һ��Ҫͬ������64���̰߳�bbox�ŵ������ڴ���ټ�������iou
	//������ VS��ʶ�𣬵���cuda��ʶ�𣬿�������

	if (tid < row_size) {
		const int cur_box_idx = threadsPerBlock * row_start + tid;
		const float* cur_box = dev_boxes + cur_box_idx * 4;
		int i = 0;
		unsigned long long int t = 0;
		int start = 0;
		if (row_start == col_start) {
			start = tid + 1; // �Խ����ϵ�����������iou
		}
		for (i = start; i < col_size; i++) {
			// ÿ���߳�Ҫ��col_size��iou
			if (devIoU(cur_box, block_boxes + i * 4, offset, iou_threshold)) {
				t |= 1ULL << i; // ���iou������ֵ��ͨ��λ���㣬tΪ64λ0 or 1����t�ĵ�iλ��Ϊ1
			}
		}
		dev_mask[cur_box_idx * gridDim.y + col_start] = t; // �޸�mask[cur_box_idx][col_start]Ϊt
	}
}
