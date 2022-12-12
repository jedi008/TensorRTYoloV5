#include "CudaOp.cuh"


#include <stdio.h>

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

cudaError_t addWithCuda2(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    fprintf(stderr, "addWithCuda2 is called.\n");

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}


__global__ void find_the_max_class_score_kernel(float* cuda_output, float* cuda_p_indexs)
{
    //int i = threadIdx.x;
    //cuda_p_indexs[0] = 0;

    int class_count = 80;
    int max_index = 0;
    for (int k = 1; k < class_count; k++)
    {
        if (cuda_output[k] > cuda_output[max_index])
        {
            max_index = k;
        }
    }
    cuda_p_indexs[0] = max_index;
}

cudaError_t find_the_max_class_score(float* cuda_output, float* class_index, unsigned int size)
{
    //fprintf(stderr, "find_the_max_class_score is called.\n");
    
    float* cuda_p_indexs;
    cudaError_t cudaStatus = cudaSuccess;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&cuda_p_indexs, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    
    
    find_the_max_class_score_kernel << <1, 1 >> > (cuda_output, cuda_p_indexs);
    //fprintf(stderr, "cuda_p_indexs[0]: %f", cuda_p_indexs[0]); //crash!!!!!!!!!!!!!!!!!!!!



    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel in .cu!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(class_index, cuda_p_indexs, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(cuda_p_indexs);
    //cudaFree(dev_a);
    //cudaFree(dev_b);

    return cudaStatus;
}


//#include <math.h>
//
//int const threadsPerBlock = sizeof(unsigned long long int) * 8; // threadsPerBlock 定义在这里，等于unsigned longlong的位数
//
//__device__ inline bool devIoU(float const* const a, float const* const b, const int offset, const float threshold) {
//    // 计算两个bbox的iou，__device__说明在cuda上执行，将被nms_cuda调用
//    float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
//    float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
//    float width = fmaxf(right - left + offset, 0.f),
//        height = fmaxf(bottom - top + offset, 0.f);
//    float interS = width * height;
//    float Sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
//    float Sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
//    return interS > threshold * (Sa + Sb - interS);
//}
//
//__global__ void nms_cuda(const int n_boxes, const float iou_threshold, const int offset, const float* dev_boxes, unsigned long long* dev_mask) {
//    // __global__表示核函数
//    const int row_start = blockIdx.y;
//    const int col_start = blockIdx.x;
//    const int tid = threadIdx.x;
//
//    if (row_start > col_start) return; // 只计算上三角的iou矩阵
//
//    const int row_size = fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
//    // 最大为threadsPerBlock，因为n_boxes可能不能被threadsPerBlock整除，获得余数
//    const int col_size = fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);
//
//    __shared__ float block_boxes[threadsPerBlock * 4];
//    // 共享内存，把同一线程块中频繁访问的64个bbox的信息放到共享内存
//    // 共享内存对同一线程块中的所有内存共享
//    // 这里每个线程，负责把一个bbox放到共享内存中 
//    if (tid < col_size) {
//        block_boxes[tid * 4 + 0] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 0];
//        // dev_boxes是一维数组的首地址
//        block_boxes[tid * 4 + 1] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 1];
//        block_boxes[tid * 4 + 2] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 2];
//        block_boxes[tid * 4 + 3] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 3];
//    }
//    __syncthreads(); // 同步！使用共享内存一定要同步，等64个线程把bbox放到共享内存后，再计算后面的iou
//    //波浪线 VS不识别，但是cuda能识别，可以无视
//
//    if (tid < row_size) {
//        const int cur_box_idx = threadsPerBlock * row_start + tid;
//        const float* cur_box = dev_boxes + cur_box_idx * 4;
//        int i = 0;
//        unsigned long long int t = 0;
//        int start = 0;
//        if (row_start == col_start) {
//            start = tid + 1; // 对角线上的跳过，不算iou
//        }
//        for (i = start; i < col_size; i++) {
//            // 每个线程要算col_size次iou
//            if (devIoU(cur_box, block_boxes + i * 4, offset, iou_threshold)) {
//                t |= 1ULL << i; // 如果iou大于阈值，通过位运算，t为64位0 or 1，把t的第i位设为1
//            }
//        }
//        dev_mask[cur_box_idx * gridDim.y + col_start] = t; // 修改mask[cur_box_idx][col_start]为t
//    }
//}
