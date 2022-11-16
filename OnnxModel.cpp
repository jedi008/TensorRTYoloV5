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
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));//nvinfer1::INetworkDefinition *network
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
    ASSERT(mOutputDims.nbDims == 3);

    return true;
}

bool OnnxModel::build_from_enginefile()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));//nvinfer1::INetworkDefinition *network
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

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();

    std::cout << "mOutputDims.nbDims: " << mOutputDims.nbDims << std::endl;
    ASSERT(mOutputDims.nbDims == 3);
    
    std::string engine_filename = "yolov5s.engine";
    std::ifstream f(engine_filename.c_str());
    if (f.good())
    {
        f.close();
        if (!load_enginefile(engine_filename)) return false;
    }
    else
    {
        if (!save_enginefile(builder, config, network, engine_filename)) return false;
    }
    
    return true;
}

bool OnnxModel::save_enginefile(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                SampleUniquePtr <nvinfer1::IBuilderConfig>& config,
                                SampleUniquePtr <nvinfer1::INetworkDefinition>& network,
                                std::string engine_filename)
{
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


    //nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);

    IHostMemory* serialized_model = mEngine->serialize();


    // 将模型序列化到engine文件中
    //const std::string engine_file_path = "yolov5s.engine";
    std::ofstream out_file(engine_filename, std::ios::binary);
    assert(out_file.is_open());
    out_file.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size()); // 写入
    out_file.close();


    return true;
}

bool OnnxModel::load_enginefile(std::string engine_filename)
{
    std::ifstream ifs(engine_filename, std::ios::binary);

    if (!ifs.is_open())
    {
        std::cout << "engine_file open fail" << std::endl;
        return false;
    }

    ifs.seekg(0, ifs.end);	// 将读指针从文件末尾开始移动0个字节
    size_t model_size = ifs.tellg();	// 返回读指针的位置，此时读指针的位置就是文件的字节数
    ifs.seekg(0, ifs.beg);	// 将读指针从文件开头开始移动0个字节
    char* modelStream = new char[model_size];
    ifs.read(modelStream, model_size);
    ifs.close();


    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());

    //nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(model_mem, model_size);
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(modelStream, model_size), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    delete runtime;

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

    clock_t start, inference_end;
    start = clock();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    inference_end = clock();
    std::cout << "inference used time = " << (double)(inference_end - start) << std::endl;//联想 26093ms  551roi 约47ms每个roi（640*640）

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


    int one_imgsize = inputC * inputH * inputH;
    for (int i = 0; i < inputN; i++)
    {
        if (i % 2 == 0)
        {
            cv::Mat img = cv::imread("cocotest_640.jpg", cv::IMREAD_COLOR);
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            cvMat2Buffer(img, hostDataBuffer + i * one_imgsize);
        }
        else
        {
            cv::Mat img2 = cv::imread("bus_640.jpg", cv::IMREAD_COLOR);
            cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
            cvMat2Buffer(img2, hostDataBuffer + i * one_imgsize);
        }

    }

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

    int onepred_size = outputBoxecount * outputBoxInfo;

    for (int i = 0; i < outputN; i++)
    {
        if (i % 2 == 0)
        {
            cv::Mat img = cv::imread("cocotest_640.jpg", cv::IMREAD_COLOR);
            predOneImage(img, output + i * onepred_size, outputBoxecount, outputBoxInfo);
        }
        else
        {
            cv::Mat img2 = cv::imread("bus_640.jpg", cv::IMREAD_COLOR);
            predOneImage(img2, output + i * onepred_size, outputBoxecount, outputBoxInfo);
        }
    }

    return true;
}
