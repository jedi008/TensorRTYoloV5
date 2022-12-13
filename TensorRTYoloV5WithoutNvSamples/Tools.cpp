#include "Tools.h"

#include "CudaOp.cuh"
#include <iostream>

void cvMat2Buffer(cv::Mat& img, float* hostDataBuffer)
{
    const int channels = img.channels();
    const int width = img.cols;
    const int height = img.rows;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                hostDataBuffer[c * width * height + h * width + w] = img.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }


    // show img data
    //std::cout << "show img data" << std::endl;
    //for (int i = 0; i < width; i++)
    //{
    //    std::cout << hostDataBuffer[i] << " ";
    //}
    //std::cout << std::endl;
}

std::vector<Object> predOneImage(cv::Mat& img, float* cuda_output, float* host_output, int outputBoxecount, int outputBoxInfo, float confidence_threshold, float nms_iou_threshold)
{
    
    std::vector<Object> proposals;
    
    
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda2(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return proposals;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return proposals;
    //}
    //return proposals;


    
    
    
    for (int i = 0; i < outputBoxecount; i++)
    {
        int indexs_size = 1 * sizeof(float);
        float* class_index = (float*)malloc(indexs_size);
        class_index[0] = 0;

        find_the_max_class_score(cuda_output + (i * outputBoxInfo + 5), class_index, indexs_size);

        int int_class_index = round(class_index[0]);
        //fprintf(stderr, "int_class_index: %d\n", int_class_index);

        float confidence = host_output[int_class_index] * host_output[i * outputBoxInfo + 4];

        if (confidence < confidence_threshold)
            continue;

        float pb_cx = host_output[i * outputBoxInfo + 0];
        float pb_cy = host_output[i * outputBoxInfo + 1];
        float pb_w = host_output[i * outputBoxInfo + 2];
        float pb_h = host_output[i * outputBoxInfo + 3];

        float x0 = pb_cx - pb_w * 0.5f;
        float y0 = pb_cy - pb_h * 0.5f;
        float x1 = pb_cx + pb_w * 0.5f;
        float y1 = pb_cy + pb_h * 0.5f;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = int_class_index;
        obj.prob = confidence;

        proposals.push_back(obj);
    }


    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_iou_threshold);

    int count = picked.size();

    std::vector<Object> objects;
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = objects[i].rect.x;
        float y0 = objects[i].rect.y;
        float x1 = objects[i].rect.x + objects[i].rect.width;
        float y1 = objects[i].rect.y + objects[i].rect.height;

        // clip
        x0 = (std::max)((std::min)(x0, (float)(640 - 1)), 0.f);
        y0 = (std::max)((std::min)(y0, (float)(640 - 1)), 0.f);
        x1 = (std::max)((std::min)(x1, (float)(640 - 1)), 0.f);
        y1 = (std::max)((std::min)(y1, (float)(640 - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    //my_draw_objects(img, objects);

    return objects;
}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

std::string draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::vector<std::string> class_names)
{
    fprintf(stderr, "draw_objects called\n");
    //static const std::vector<const char*> class_names_default = {
//"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//"hair drier", "toothbrush"
//};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
            obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(0, 0, 255), 4);

        char text[256];
        sprintf_s(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    //QString image_path = QString("%1/cache/cache_bg_img-%2.jpg").arg(QApplication::applicationDirPath()).arg(time.toString("HH-mm-ss.zzz"));
    std::string image_path = "./outimg.jpg";
    cv::imwrite(image_path, image);
    cv::imshow("image", image);
    cv::waitKey(0);

    return image_path;
}

