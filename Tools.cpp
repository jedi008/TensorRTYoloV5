#include "Tools.h"

void cvMat2Buffer(cv::Mat& img, float* hostDataBuffer)
{
    const int channels = img.channels();
    const int width = img.cols;
    const int height = img.rows;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                hostDataBuffer[c * width * height + h * width + w] =
                    img.at<cv::Vec3b>(h, w)[c] / 255.0f;
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

void predOneImage(cv::Mat& img, float* output, int outputBoxecount, int outputBoxInfo)
{
    std::vector<Object> proposals;
    const float confidence_threshold = 0.35;
    const float nms_threshold = 0.45;
    for (int i = 0; i < outputBoxecount; i++)
    {
        // find class index with max class score
        float class_score = -FLT_MAX;
        int class_index = 0;
        for (int k = 0; k < outputBoxInfo - 5; k++)
        {
            float score = output[i * outputBoxInfo + 5 + k];
            if (score > class_score)
            {
                class_index = k;
                class_score = score;
            }
        }
        float confidence = class_score * output[i * outputBoxInfo + 4];

        if (confidence < confidence_threshold)
            continue;

        float pb_cx = output[i * outputBoxInfo + 0];
        float pb_cy = output[i * outputBoxInfo + 1];
        float pb_w = output[i * outputBoxInfo + 2];
        float pb_h = output[i * outputBoxInfo + 3];

        float x0 = pb_cx - pb_w * 0.5f;
        float y0 = pb_cy - pb_h * 0.5f;
        float x1 = pb_cx + pb_w * 0.5f;
        float y1 = pb_cy + pb_h * 0.5f;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = class_index;
        obj.prob = confidence;

        proposals.push_back(obj);
    }


    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

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
    
    my_draw_objects(img, objects);
}
