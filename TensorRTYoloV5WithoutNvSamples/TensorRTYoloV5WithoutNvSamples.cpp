﻿// TensorRTYoloV5WithoutNvSamples.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>

#include "YoloV5Model.h"

int main()
{
    for (unsigned long i = 0; i < ULONG_MAX; i++)
    {
        float fa = i;
        unsigned long ia = fa;
        if (ia != i)
        {
            std::cout << "fa : " << fa << "  ia: " << ia << std::endl;
            break;
        }
    }

    cudaSetDevice(0);
    
    YoloV5Model* yolov5_model = new YoloV5Model();
    yolov5_model->build_from_enginefile("yolov5s.onnx");

    std::vector<cv::Mat > images;
    cv::Mat img = cv::imread("cocotest_640.jpg", cv::IMREAD_COLOR);
    images.push_back(img);

    cv::Mat img2 = cv::imread("bus_640.jpg", cv::IMREAD_COLOR);
    images.push_back(img2);

    
    clock_t start, inference_end;
    start = clock();

    yolov5_model->infer(images);

    inference_end = clock();
    std::cout << "inference rois used time = " << (double)(inference_end - start) << std::endl;//first time: 4356ms

    
    std::cout << "Hello World!\n";
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
