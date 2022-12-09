#pragma once

#include "./common/argsParser.h"
#include "./common/buffers.h"
#include "./common/common.h"
#include "./common/logger.h"
#include "./common/parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

using samplesCommon::SampleUniquePtr;

class OnnxModel
{
public:
    OnnxModel(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Function builds the  engine
    //!
    bool build_from_enginefile();
    bool save_enginefile(SampleUniquePtr<nvinfer1::IBuilder>& builder, 
                         SampleUniquePtr <nvinfer1::IBuilderConfig>& config, 
                         SampleUniquePtr <nvinfer1::INetworkDefinition>& network,
                         std::string engine_filename);
    bool load_enginefile(std::string engine_filename);

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{ 0 };             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool yolo_verifyOutput(const samplesCommon::BufferManager& buffers);
};

