#pragma once

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include <cuda_runtime_api.h>
#include "NvOnnxParser.h"

#include <string>
#include <iostream>


#include "common.h"

class YoloV5Model
{
public:
	YoloV5Model();

	bool build_from_enginefile(std::string filename);

private:
    bool constructNetwork(  SampleUniquePtr<nvinfer1::IBuilder>& builder,
                            SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                            SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                            SampleUniquePtr<nvonnxparser::IParser>& parser);

    bool save_enginefile(   SampleUniquePtr<nvinfer1::IBuilder>& builder,
                            SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                            SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                            std::string engine_filename);

    bool load_enginefile(std::string engine_filename);

private:
    MyLogger logger;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
};

