#include "OnnxModel.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Tools.h"

bool OnnxModel::build()
{
    std::cout << "step================================b0" << std::endl;
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    std::cout << "step================================b1" << std::endl;
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    std::cout << "step================================b2" << std::endl;
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    std::cout << "step================================b3" << std::endl;
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    std::cout << "step================================b4" << std::endl;
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    std::cout << "step================================b5" << std::endl;
    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);


    std::cout << "step================================b6" << std::endl;
    SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
    if (!plan)
    {
        return false;
    }

    std::cout << "step================================b7" << std::endl;
    SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
    if (!runtime)
    {
        return false;
    }

    std::cout << "step================================b8" << std::endl;
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


    cv::Mat img = cv::imread("D:/TestData/cocotest_640.jpg", cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
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
    std::cout << "show img data" << std::endl;
    for (int i = 0; i < width; i++)
    {
        std::cout << hostDataBuffer[i] << " ";
    }
    std::cout << std::endl;

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

    const int outputSize = outputN * outputBoxecount * outputBoxInfo;
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


    cv::Mat img = cv::imread("D:/TestData/cocotest_640.jpg", cv::IMREAD_COLOR);
    my_draw_objects(img, objects);

    return true;
}
