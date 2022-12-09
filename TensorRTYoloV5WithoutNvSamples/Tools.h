#pragma once

#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


struct Object
{
    cv::Rect_<float> rect;
    int label{ 0 };
    float prob{ 0.f };
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void cvMat2Buffer(cv::Mat& img, float* hostDataBuffer);

std::vector<Object> predOneImage(cv::Mat& img, float* output, int outputBoxecount, int outputBoxInfo, float confidence_threshold, float nms_iou_threshold);

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);

void qsort_descent_inplace(std::vector<Object>& faceobjects);

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false);

std::string draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::vector<std::string> class_names);