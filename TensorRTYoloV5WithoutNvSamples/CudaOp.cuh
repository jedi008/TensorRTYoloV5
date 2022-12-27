#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "NvInferRuntime.h"
#include <cuda_runtime_api.h>

#include "common.cuh"



int cuda_after_op_oneimg(float* cuda_output, int outputBoxecount, int output_box_size, float** host_objects_p, float confidence_threshold, float nms_iou_threshold);