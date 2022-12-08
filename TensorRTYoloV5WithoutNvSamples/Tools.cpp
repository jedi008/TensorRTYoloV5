#include "Tools.h"

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
