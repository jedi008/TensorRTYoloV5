#include "CudaOp.cuh"



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


__global__ void find_all_max_class_score_kernel(float* cuda_output, int output_box_size, int* cuda_objects_index, int* cuda_objects_index_mask, int outputBoxecount)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= outputBoxecount) return;

    int class_count = output_box_size - 5;//=80
    int max_index = 0;

    float* base_p = cuda_output + n * output_box_size + 5;

    for (int k = 1; k < class_count; k++)
    {
        if (base_p[k] > base_p[max_index])
        {
            max_index = k;
        }
    }
    cuda_objects_index[n] = max_index;

    
    float confidence = base_p[max_index] * cuda_output[n * output_box_size + 4];

    static const float confidence_threshold = 0.45;

    cuda_objects_index_mask[n] = confidence < confidence_threshold ? 0 : 1;
}

__global__ void init_objects_kernel(float* cuda_output, int output_box_size, float* cuda_objects, int* cuda_objects_index, int* cuda_objects_index_mask, int outputBoxecount)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= outputBoxecount) return;

    //float confidence = cuda_output[n * output_box_size + 5 + max_index] * cuda_output[n * output_box_size + 4];

    //for (int i = 0; i < outputBoxecount; i++)
//{
//    int indexs_size = 1 * sizeof(float);
//    float* class_index = (float*)malloc(indexs_size);
//    class_index[0] = 0;

//    find_onebox_max_class_score(cuda_output + (i * outputBoxInfo + 5), class_index, indexs_size);

//    int int_class_index = round(class_index[0]);
//    //fprintf(stderr, "int_class_index: %d\n", int_class_index);

//    float confidence = host_output[int_class_index] * host_output[i * outputBoxInfo + 4];

//    if (confidence < confidence_threshold)
//        continue;

//    float pb_cx = host_output[i * outputBoxInfo + 0];
//    float pb_cy = host_output[i * outputBoxInfo + 1];
//    float pb_w = host_output[i * outputBoxInfo + 2];
//    float pb_h = host_output[i * outputBoxInfo + 3];

//    float x0 = pb_cx - pb_w * 0.5f;
//    float y0 = pb_cy - pb_h * 0.5f;
//    float x1 = pb_cx + pb_w * 0.5f;
//    float y1 = pb_cy + pb_h * 0.5f;

//    Object obj;
//    obj.rect.x = x0;
//    obj.rect.y = y0;
//    obj.rect.width = x1 - x0;
//    obj.rect.height = y1 - y0;
//    obj.label = int_class_index;
//    obj.prob = confidence;

//    proposals.push_back(obj);
//}
}

//后期再将其优化为并行模式，逐步两两相加
__global__ void array_sum_kernel(int* cuda_array, int array_size)//该cuda_array 为定制化的，申请空间大小时为array_size + 1, 最后一位用于存放sum
{
    cuda_array[array_size] = 0;
    for (int i = 0; i < array_size; i++)
    {
        cuda_array[array_size] += cuda_array[i];
    }

    printf("array_sum run successful, res: %d\n", cuda_array[array_size]);
}

void find_all_max_class_score(float* cuda_output, int outputBoxecount)
{
    printf("find_all_max_class_score called.\n");
    int* cuda_objects_index;
    int* cuda_objects_index_mask;
    float* cuda_objects;

    //HANDLE_ERROR(cudaSetDevice(0));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //cudaEventRecord(start);
    ////你要测试的kernel
    //wait until the stop event completes
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float time;
    //cudaEventElapsedTime(&time, start, stop);

    // Allocate GPU buffers.
    HANDLE_ERROR(cudaMalloc((void**)&cuda_objects_index, outputBoxecount * sizeof(int)) );
    HANDLE_ERROR(cudaMalloc((void**)&cuda_objects_index_mask, (1 + outputBoxecount) * sizeof(int)));//多给一个size用于存放array的和
    
    
    int output_box_size = 85;
    int grid_size = (outputBoxecount + 512) / 1024;//outputBoxecount: 25200
    cudaEventRecord(start);
    find_all_max_class_score_kernel<<<grid_size, 1024>>>(cuda_output, output_box_size, cuda_objects_index, cuda_objects_index_mask, outputBoxecount);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("find_all_max_class_score_kernel used %fms\n",time);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());



    array_sum_kernel << <1, 1 >> > (cuda_objects_index_mask, outputBoxecount);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());


    int objects_count = 0;
    HANDLE_ERROR(cudaMemcpy(&objects_count, cuda_objects_index_mask + outputBoxecount, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMalloc((void**)&cuda_objects, 6 * objects_count * sizeof(float)));//每个Boxinfo x,y,w,h,label,prob 6个值

    //init_objects_kernel << <grid_size, 1024 >> > (cuda_output, output_box_size, cuda_objects, cuda_objects_index, cuda_objects_index_mask, outputBoxecount);


    cudaFree(cuda_objects_index);
    cudaFree(cuda_objects_index_mask);
    cudaFree(cuda_objects);
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
