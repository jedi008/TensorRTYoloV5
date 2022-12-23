#include "CudaOp.cuh"

#include <malloc.h>

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
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

    
    printf("address c: %x\n", c);
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

//blocksize 必须设置为2的整数次方
__global__ void kernel_find_all_max_class_score(float* cuda_output, int output_box_size, int* cuda_objects_index, int* cuda_objects_index_mask,  int class_count)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int len = blockDim.x;
    const int gridsize = gridDim.x;
    const int n = bid * len + tid;
    
    __shared__ int index[128];
    index[tid] = (tid < class_count) ? tid : 0;
    __syncthreads();

    if (tid >= class_count) return;

    float* base_p = cuda_output + bid * output_box_size + 5;

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            if (base_p[index[tid]] < base_p[index[tid + offset]])
            {
                index[tid] = index[tid + offset];
            }
            __syncthreads();
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            if (base_p[index[tid]] < base_p[index[tid + offset]])
            {
                index[tid] = index[tid + offset];
            }
            __syncwarp();
        }
    }
    
    if (tid == 0)
    {
        cuda_objects_index[bid] = index[0];

        const float confidence_threshold = 0.45;
        if ((base_p[index[0]] * cuda_output[bid * output_box_size + 4]) > confidence_threshold)
        {
            cuda_objects_index_mask[bid] = 1;
            atomicAdd(cuda_objects_index_mask + gridsize, 1);
        }
        else
        {
            cuda_objects_index_mask[bid] = 0;
        }
    }
}

__global__ void kernel_init_objects(float* cuda_output, int output_box_size, float* cuda_objects, int* cuda_objects_index, int* cuda_objects_index_mask, int output_box_count)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    if (n >= output_box_count) return;

    if (!cuda_objects_index_mask[n]) return;

    int object_index = atomicAdd(cuda_objects_index_mask + output_box_count, -1) - 1;

    float* cuda_object_basep = cuda_objects + object_index * 6;
    float* cuda_output_basep = cuda_output + n * output_box_size;
    
    float pb_cx = cuda_output_basep[0];
    float pb_cy = cuda_output_basep[1];
    float pb_w = cuda_output_basep[2];
    float pb_h = cuda_output_basep[3];

    float x0 = pb_cx - pb_w * 0.5f;
    float y0 = pb_cy - pb_h * 0.5f;

    cuda_object_basep[0] = x0;
    cuda_object_basep[1] = y0;
    cuda_object_basep[2] = pb_w;
    cuda_object_basep[3] = pb_h;
    cuda_object_basep[4] = float(cuda_objects_index[n]);
    cuda_object_basep[5] = cuda_output_basep[5 + cuda_objects_index[n]] * cuda_output_basep[4];
}


__global__ void kernel_objects_sort(float* cuda_objects, int objects_count, float* cuda_objects_sorted)
{
    int tid = threadIdx.x;

    float* cuda_objects_bp = cuda_objects + tid * 6;
    float prob = cuda_objects_bp[5];
    extern __shared__ float cache[];
    cache[tid] = prob;
    __syncthreads();

    //统计比当前置信度大的有几个
    int count = 0;
    for (int i = 0; i < objects_count; i++)
    {
        if (prob < cache[i])
        {
            count++;
        }
    }

    float* cuda_objects_sorted_bp = cuda_objects_sorted + count * 6;
    cuda_objects_sorted_bp[0] = cuda_objects_bp[0];
    cuda_objects_sorted_bp[1] = cuda_objects_bp[1];
    cuda_objects_sorted_bp[2] = cuda_objects_bp[2];
    cuda_objects_sorted_bp[3] = cuda_objects_bp[3];
    cuda_objects_sorted_bp[4] = cuda_objects_bp[4];
    cuda_objects_sorted_bp[5] = prob;
    //cudaMemcpy(cuda_objects_sorted + count * 6, cuda_objects + tid * 6, 6 * sizeof(float), cudaMemcpyDeviceToDevice); //不可在核函数中运行
}

int find_all_max_class_score(float* cuda_output, int output_box_count, float** host_objects_p)
{
    printf("find_all_max_class_score called.\n");
    int* cuda_objects_index;
    int* cuda_objects_index_mask;
    float* cuda_objects;

    //HANDLE_ERROR(cudaSetDevice(0));

    float elapsed_time;
    cudaEvent_t start, step1, step2, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&step1);
    cudaEventCreate(&step2);
    cudaEventCreate(&stop);


    // Allocate GPU buffers.
    HANDLE_ERROR(cudaMalloc((void**)&cuda_objects_index, output_box_count * sizeof(int)) );
    HANDLE_ERROR(cudaMalloc((void**)&cuda_objects_index_mask, (1 + output_box_count) * sizeof(int)));//多给一个size用于存放array的和
    
    
    int output_box_size = 85;
    int grid_size = output_box_count;//outputBoxecount: 25200
    cudaEventRecord(start);
    
    kernel_find_all_max_class_score << <grid_size, 128 >> > (cuda_output, output_box_size, cuda_objects_index, cuda_objects_index_mask, output_box_size - 5);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());

    
    cudaEventRecord(step1);
    cudaEventSynchronize(step1);
    cudaEventElapsedTime(&elapsed_time, start, step1);
    printf("find_all_max_class_score_kernel used %fms\n", elapsed_time);


    int objects_count = 0;
    HANDLE_ERROR(cudaMemcpy(&objects_count, cuda_objects_index_mask + output_box_count, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMalloc((void**)&cuda_objects, 6 * objects_count * sizeof(float)));//每个Boxinfo x,y,w,h,label,prob 6个值
    printf("objects_count: %d\n", objects_count);
    
    cudaEventRecord(step2);
    cudaEventSynchronize(step2);
    cudaEventElapsedTime(&elapsed_time, step1, step2);
    printf("cudaMalloc cuda_objects used %fms\n", elapsed_time);



    //可以提前开辟100个objects，将该步骤与find_all_max_class_score_kernel 合并
    kernel_init_objects << <(output_box_count + 1023)/1024, 1024 >> > (cuda_output, output_box_size, cuda_objects, cuda_objects_index, cuda_objects_index_mask, output_box_count);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());



    float* cuda_objects_sorted;
    HANDLE_ERROR(cudaMalloc((void**)&cuda_objects_sorted, 6 * objects_count * sizeof(float)));//每个Boxinfo x,y,w,h,label,prob 6个值
    kernel_objects_sort << <1, objects_count, objects_count * sizeof(float) >> > (cuda_objects, objects_count, cuda_objects_sorted);//默认objects_count不会超过1024个
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());



    float* d_host_objects = (float*)malloc(6 * objects_count * sizeof(float));
    HANDLE_ERROR(cudaMemcpy(d_host_objects, cuda_objects_sorted, 6 * objects_count * sizeof(float), cudaMemcpyDeviceToHost));
    //printf("1 d_host_objects: %f - %f - %f - %f - %f - %f\n", d_host_objects[0], d_host_objects[1], d_host_objects[2], d_host_objects[3], d_host_objects[4], d_host_objects[5]);
    *host_objects_p = d_host_objects;

    cudaFree(cuda_objects_index);
    cudaFree(cuda_objects_index_mask);
    cudaFree(cuda_objects);
    cudaFree(cuda_objects_sorted);

    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("all cuda op used %fms\n", elapsed_time);

    return objects_count;
}


