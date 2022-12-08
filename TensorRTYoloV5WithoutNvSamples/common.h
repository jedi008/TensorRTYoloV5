#pragma once

#include "NvInfer.h"
//#include "NvInferRuntime.h"
//#include <cuda_runtime_api.h>
//#include "NvOnnxParser.h"

//#ifdef _MSC_VER
//#include ".\windows\getopt.h"
//#endif

#undef ASSERT
#define ASSERT(condition)                                                   \
    do                                                                      \
    {                                                                       \
        if (!(condition))                                                   \
        {                                                                   \
            std::cerr << "Assertion failure: " << #condition << std::endl;  \
            abort();                                                        \
        }                                                                   \
    } while (0)


struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

static auto StreamDeleter = [](cudaStream_t* pStream)
{
    if (pStream)
    {
        cudaStreamDestroy(*pStream);
        delete pStream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}


class MyLogger : public nvinfer1::ILogger {
public:
    explicit MyLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
        : severity_(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= severity_) {
            std::cerr << msg << std::endl;
        }
    }
    nvinfer1::ILogger::Severity severity_;
};