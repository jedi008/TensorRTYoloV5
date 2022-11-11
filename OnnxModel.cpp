#include "OnnxModel.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Tools.h"

bool OnnxModel::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();

    std::cout << "mOutputDims.nbDims: " << mOutputDims.nbDims << std::endl;
    //ASSERT(mOutputDims.nbDims == 2);

    return true;
}

bool OnnxModel::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!yolo_verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

bool OnnxModel::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

bool OnnxModel::processInput(const samplesCommon::BufferManager& buffers)
{
    std::cout << "\nmInputDims.nbDims: " << mInputDims.nbDims << std::endl;
    std::cout << "\nmInputDims.d[0]: " << mInputDims.d[0] << std::endl;
    std::cout << "\nmInputDims.d[1]: " << mInputDims.d[1] << std::endl;
    std::cout << "\nmInputDims.d[2]: " << mInputDims.d[2] << std::endl;
    std::cout << "\nmInputDims.d[3]: " << mInputDims.d[3] << std::endl;

    const int inputN = mInputDims.d[0]; //batch_size
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];


    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

    cv::Mat img = cv::imread("D:/TestData/cocotest_640.jpg", cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cvMat2Buffer(img, hostDataBuffer);

    cv::Mat img2 = cv::imread("D:/TestData/bus_640.jpg", cv::IMREAD_COLOR);
    cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
    cvMat2Buffer(img2, hostDataBuffer + inputC * inputH * inputH);

    return true;
}

bool OnnxModel::yolo_verifyOutput(const samplesCommon::BufferManager& buffers)
{
    std::cout << "\nmOutputDims.nbDims: " << mOutputDims.nbDims << std::endl;
    std::cout << "\nmOutputDims.d[0]: " << mOutputDims.d[0] << std::endl;
    std::cout << "\nmOutputDims.d[1]: " << mOutputDims.d[1] << std::endl;
    std::cout << "\nmOutputDims.d[2]: " << mOutputDims.d[2] << std::endl;

    const int outputN = mOutputDims.d[0]; //batch_size
    const int outputBoxecount = mOutputDims.d[1];
    const int outputBoxInfo = mOutputDims.d[2];

    //const int outputSize = outputN * outputBoxecount * outputBoxInfo;
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

    for (int i = 0; i < outputBoxecount; i += outputBoxInfo)
    {
        for (int j = 0; j < outputBoxInfo; j++)
        {
            std::cout << output[j] << " ";
        }
        std::cout << std::endl;

        break;
    }

    cv::Mat img = cv::imread("D:/TestData/cocotest_640.jpg", cv::IMREAD_COLOR);
    predOneImage(img, output, outputBoxecount, outputBoxInfo);

    cv::Mat img2 = cv::imread("D:/TestData/bus_640.jpg", cv::IMREAD_COLOR);
    predOneImage(img2, output + outputBoxecount * outputBoxInfo, outputBoxecount, outputBoxInfo);

    return true;
}
