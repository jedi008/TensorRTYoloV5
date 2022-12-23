#include "YoloV5Model.h"

#include <iostream>
#include <fstream>

#include "Tools.h"

#include "CudaOp.cuh"

YoloV5Model::YoloV5Model()
{
}

bool YoloV5Model::build_from_enginefile(std::string onnxfilename)
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
	if (!builder)
	{
		return false;
	}

	const uint32_t explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
	if (!network)
	{
		return false;
	}

	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		return false;
	}
	//config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 25);//其中最重要的一个属性是工作空间的最大容量。在网络层实现过程中通常会需要一些临时的工作空间，这个属性会限制最大能申请的工作空间的容量
	//if (builder->platformHasFastFp16()) {
	//	config->setFlag(nvinfer1::BuilderFlag::kFP16);
	//}

	
	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
	if (!parser)
	{
		return false;
	}
	parser->parseFromFile(onnxfilename.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));
	// 如果有错误则输出错误信息
	for (int32_t i = 0; i < parser->getNbErrors(); ++i) 
	{
		std::cout << parser->getError(i)->desc() << std::endl;
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

bool YoloV5Model::infer(std::vector<cv::Mat > images)
{
	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)
	{
		return false;
	}
	
	std::cout << "\nmInputDims.nbDims: " << mInputDims.nbDims << std::endl;
	std::cout << "\nmInputDims.d[0]: " << mInputDims.d[0] << std::endl;
	std::cout << "\nmInputDims.d[1]: " << mInputDims.d[1] << std::endl;
	std::cout << "\nmInputDims.d[2]: " << mInputDims.d[2] << std::endl;
	std::cout << "\nmInputDims.d[3]: " << mInputDims.d[3] << std::endl;

	void* buffers[2];
	// 获取模型输入尺寸并分配GPU内存
	const int inputBatch_size = mInputDims.d[0];
	if (inputBatch_size < images.size()) 
		return false;
	const int inputC = mInputDims.d[1];
	const int inputH = mInputDims.d[2];
	const int inputW = mInputDims.d[3];
	int one_imgsize = inputC * inputH * inputW;
	int input_size = one_imgsize * inputBatch_size * sizeof(float);
	cudaMalloc(&buffers[0], input_size);

	// 获取模型输出尺寸并分配GPU内存
	const int outputBatch_size = mOutputDims.d[0]; //batch_size
	const int outputBoxecount = mOutputDims.d[1];
	const int outputBoxInfo = mOutputDims.d[2];
	//int output_size = 1;
	//for (int j = 0; j < output_dim.nbDims; ++j) {
	//	output_size *= output_dim.d[j];
	//}
	int output_size = outputBatch_size * outputBoxecount * outputBoxInfo * sizeof(float);
	cudaMalloc(&buffers[1], output_size);

	
	// 给模型输出数据分配相应的CPU内存
	float* host_input_buffer = (float*)malloc(input_size);
	float* host_output_buffer = (float*)malloc(output_size);

	for (int i = 0; i < images.size(); i++)
	{
		cv::Mat img = images[i];

		cv::Mat img_input;
		cv::cvtColor(img, img_input, cv::COLOR_BGR2RGB);
		cvMat2Buffer(img_input, host_input_buffer + i * one_imgsize);
	}

	//copyInputToDevice
	CHECK(cudaMemcpy(buffers[0], host_input_buffer, input_size, cudaMemcpyHostToDevice));
	
	bool status = context->executeV2(buffers);
	if (!status)
	{
		return false;
	}

	//copyOutputToHost
	//CHECK(cudaMemcpy(host_output_buffer, buffers[1], output_size, cudaMemcpyDeviceToHost));


	int onepred_size = outputBoxecount * outputBoxInfo;
	std::vector<Object> proposals;
	for (int i = 0; i < images.size(); i++)
	{
		cv::Mat img = images[i];
		std::vector<Object> objects = predOneImage(	img, 
													(float*)buffers[1] + i * onepred_size,
													outputBoxecount, 
													outputBoxInfo, 
													0.45, 
													0.35);


		//图块上检测结果转换为大图上的坐标
		//for (int j = 0; j < objects.size(); j++)
		//{
		//	objects[j].rect.x += pads.at(i).x;
		//	objects[j].rect.y += pads.at(i).y;
		//}

		//proposals.insert(proposals.end(), objects.begin(), objects.end());

		static const std::vector<std::string> class_names_default = {
			"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
			"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
			"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
			"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
			"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
			"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
			"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
			"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
			"hair drier", "toothbrush"
		};
		//draw_objects(img, objects, class_names_default);
	}

	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
	free(host_input_buffer);
	free(host_output_buffer);
	
	return true;
}

bool YoloV5Model::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	//auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
	//	static_cast<int>(sample::gLogger.getReportableSeverity()));
	//if (!parsed)
	//{
	//	return false;
	//}

	//if (mParams.fp16)
	//{
	//	config->setFlag(BuilderFlag::kFP16);
	//}
	//if (mParams.int8)
	//{
	//	config->setFlag(BuilderFlag::kINT8);
	//	samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
	//}

	//samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

	return true;
}

bool YoloV5Model::save_enginefile(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, std::string engine_filename)
{
	// CUDA stream used for profiling by the builder.
	auto profileStream = makeCudaStream();
	if (!profileStream)
	{
		return false;
	}
	config->setProfileStream(*profileStream);


	SampleUniquePtr<nvinfer1::IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
	if (!plan)
	{
		return false;
	}

	SampleUniquePtr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(logger) };
	if (!runtime)
	{
		return false;
	}

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
	if (!mEngine)
	{
		return false;
	}


	//nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);

	nvinfer1::IHostMemory* serialized_model = mEngine->serialize();


	// 将模型序列化到engine文件中
	//const std::string engine_file_path = "yolov5s.engine";
	std::ofstream out_file(engine_filename, std::ios::binary);
	ASSERT(out_file.is_open());
	out_file.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size()); // 写入
	out_file.close();


	return true;
}

bool YoloV5Model::load_enginefile(std::string engine_filename)
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


	SampleUniquePtr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(logger) };
	if (!runtime)
	{
		return false;
	}

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelStream, model_size), InferDeleter());
	if (!mEngine)
	{
		return false;
	}

	return true;
}
