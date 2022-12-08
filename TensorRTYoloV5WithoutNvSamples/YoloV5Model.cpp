#include "YoloV5Model.h"

#include <iostream>
#include <fstream>

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
	nvinfer1::Dims mInputDims = network->getInput(0)->getDimensions();
	ASSERT(mInputDims.nbDims == 4);

	ASSERT(network->getNbOutputs() == 1);
	nvinfer1::Dims mOutputDims = network->getOutput(0)->getDimensions();

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
