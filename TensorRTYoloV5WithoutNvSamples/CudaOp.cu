#include "CudaOp.cuh"

#include <malloc.h>
#include <math.h>


//blocksize ��������Ϊ2�������η�
__global__ void kernel_find_all_max_class_score(float* cuda_output, int output_box_size, int* cuda_objects_index, int* cuda_objects_index_mask,  int class_count, float confidence_threshold)
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

    //ͳ�Ʊȵ�ǰ���Ŷȴ���м���
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
    //cudaMemcpy(cuda_objects_sorted + count * 6, cuda_objects + tid * 6, 6 * sizeof(float), cudaMemcpyDeviceToDevice); //�����ں˺���������
}


__global__ void kernel_nms(float* cuda_objects_sorted, int objects_count, bool* cuda_pickedmask, float nms_threshold, bool agnostic = false)
{
    const int tid = threadIdx.x;

    cuda_pickedmask[tid] = true;

    float* check_box_p = cuda_objects_sorted + 6 * tid;
    extern __shared__ float areas[];
    areas[tid] = check_box_p[2] * check_box_p[3];//area
    __syncthreads();


    for (int i = 0; i < objects_count; i++)
    {
        if (tid < i && cuda_pickedmask[tid])//���ǵ�i��box�Ƿ���Ҫ����picked�� 1.�Լ����Լ����ü���iou 2.���ں����box score������С�� ���������nmsɸ����Ч 3.֮ǰ�ļ������Ѿ���pass����BoxҲ�����ٿ���
        {
            float* box_p = cuda_objects_sorted + 6 * i;//�����Ƿ���Ҫͨ��nms����box
            if (agnostic || fabsf(check_box_p[4] - box_p[4]) < 0.1)//����ͬһ���������岻��nms. ���������������߾���ͬһ����box����������nms����
            {
                float x1 = fmaxf(box_p[0], check_box_p[0]);
                float y1 = fmaxf(box_p[1], check_box_p[1]);
                float x2 = fminf(box_p[0] + box_p[2], check_box_p[0] + check_box_p[2]);
                float y2 = fminf(box_p[1] + box_p[3], check_box_p[1] + check_box_p[3]);
                float width = fmaxf(x2 - x1, 0);
                float height = fmaxf(y2 - y1, 0);
                float inter_area = width * height;
                float union_area = areas[i] + areas[tid] - inter_area;
                if (inter_area / union_area > nms_threshold)
                {
                    cuda_pickedmask[i] = false;
                    //printf("i tid del one: %d-%d %f %f %f %f %f %f\n", i, tid, box_p[0], box_p[1], box_p[2], box_p[3], box_p[4], box_p[5]);
                }
            }
        }
        __syncthreads();
    }
}


int cuda_after_op_oneimg(float* cuda_output, int output_box_count, float** host_objects_p, float confidence_threshold, float nms_iou_threshold)
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
    HANDLE_ERROR(cudaMalloc((void**)&cuda_objects_index_mask, (1 + output_box_count) * sizeof(int)));//���һ��size���ڴ��array�ĺ�
    
    
    int output_box_size = 85;
    int grid_size = output_box_count;//outputBoxecount: 25200
    cudaEventRecord(start);
    
    kernel_find_all_max_class_score << <grid_size, 128 >> > (cuda_output, output_box_size, cuda_objects_index, cuda_objects_index_mask, output_box_size - 5, confidence_threshold);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());

    
    cudaEventRecord(step1);
    cudaEventSynchronize(step1);
    cudaEventElapsedTime(&elapsed_time, start, step1);
    printf("find_all_max_class_score_kernel used %fms\n", elapsed_time);


    int objects_count = 0;
    HANDLE_ERROR(cudaMemcpy(&objects_count, cuda_objects_index_mask + output_box_count, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMalloc((void**)&cuda_objects, 6 * objects_count * sizeof(float)));//ÿ��Boxinfo x,y,w,h,label,prob 6��ֵ
    printf("objects_count: %d\n", objects_count);
    
    cudaEventRecord(step2);
    cudaEventSynchronize(step2);
    cudaEventElapsedTime(&elapsed_time, step1, step2);
    printf("cudaMalloc cuda_objects used %fms\n", elapsed_time);



    //������ǰ����100��objects�����ò�����find_all_max_class_score_kernel �ϲ�
    kernel_init_objects << <(output_box_count + 1023)/1024, 1024 >> > (cuda_output, output_box_size, cuda_objects, cuda_objects_index, cuda_objects_index_mask, output_box_count);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());
    if (objects_count == 0) //û�м�⵽һ��Ŀ���ֱ�ӷ���
    {
        cudaFree(cuda_objects_index);
        cudaFree(cuda_objects_index_mask);
        cudaFree(cuda_objects);
        return 0;
    }



    float* cuda_objects_sorted;
    HANDLE_ERROR(cudaMalloc((void**)&cuda_objects_sorted, 6 * objects_count * sizeof(float)));//ÿ��Boxinfo x,y,w,h,label,prob 6��ֵ
    kernel_objects_sort << <1, objects_count, objects_count * sizeof(float) >> > (cuda_objects, objects_count, cuda_objects_sorted);//Ĭ��objects_count���ᳬ��1024��
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());


    //cuda nms
    bool* cuda_pickedmask;
    HANDLE_ERROR(cudaMalloc((void**)&cuda_pickedmask, objects_count * sizeof(bool)));
    kernel_nms << <1, objects_count, objects_count * sizeof(float)>> > (cuda_objects_sorted, objects_count, cuda_pickedmask, nms_iou_threshold);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());


    bool* host_pickedmask = (bool*)malloc(objects_count * sizeof(bool));
    HANDLE_ERROR(cudaMemcpy(host_pickedmask, cuda_pickedmask, objects_count * sizeof(bool), cudaMemcpyDeviceToHost));
    int picked_objects_count = 0;
    for (int i = 0; i < objects_count; i++)
    {
        if (host_pickedmask[i]) ++picked_objects_count;
    }

    printf("picked_objects_count: %d\n", picked_objects_count);
    float* d_host_objects = (float*)malloc(6 * picked_objects_count * sizeof(float));
    int picked_index = 0;
    for (int i = 0; i < objects_count; i++)
    {
        if (host_pickedmask[i])
        { 
            HANDLE_ERROR(cudaMemcpy(d_host_objects + picked_index * 6, cuda_objects_sorted + i * 6, 6 * sizeof(float), cudaMemcpyDeviceToHost));
            ++picked_index;
        }
            
    }
    //HANDLE_ERROR(cudaMemcpy(d_host_objects, cuda_objects_sorted, 6 * objects_count * sizeof(float), cudaMemcpyDeviceToHost));
    //printf("1 d_host_objects: %f - %f - %f - %f - %f - %f\n", d_host_objects[0], d_host_objects[1], d_host_objects[2], d_host_objects[3], d_host_objects[4], d_host_objects[5]);
    *host_objects_p = d_host_objects;

    cudaFree(cuda_objects_index);
    cudaFree(cuda_objects_index_mask);
    cudaFree(cuda_objects);
    cudaFree(cuda_objects_sorted);
    cudaFree(cuda_pickedmask);

    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("all cuda op used %fms\n", elapsed_time);

    return picked_objects_count;
}


