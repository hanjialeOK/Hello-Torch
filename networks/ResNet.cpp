#include "ResNet.h"

/* BottleNeck */

BottleNeckImpl::BottleNeckImpl(int64_t inplanes, int64_t planes, int64_t stride,
    torch::nn::Sequential downsample_) : 
    downsample(downsample_),
    conv1(torch::nn::Conv2dOptions(inplanes, planes, 1)
            .stride(1).padding(0).bias(false)),
    bn1(planes),
    conv2(torch::nn::Conv2dOptions(planes, planes, 3)
            .stride(stride).padding(1).bias(false)),
    bn2(planes),
    conv3(torch::nn::Conv2dOptions(planes, planes * expansion, 1)
            .stride(1).padding(0).bias(false)),
    bn3(planes * expansion)
{
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);

    if (!downsample->is_empty()) {
        register_module("downsample", downsample);
    }
}

torch::Tensor BottleNeckImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x.clone();

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = bn2->forward(x);
    x = torch::relu(x);

    x = conv3->forward(x);
    x = bn3->forward(x);

    if (!downsample->is_empty()){
        residual = downsample->forward(residual);
    }
    x += residual;
    x = torch::relu(x);

    return x;
}

/* ResNet */
ResNetImpl::ResNetImpl(std::vector<int> layers, int num_classes) : 
    conv1(torch::nn::Conv2dOptions(3, 64, 7)
            .stride(2).padding(3).bias(false)),
    bn1(64),
    layer1(_make_layer(64,  layers[0])),
    layer2(_make_layer(128, layers[1], 2)),
    layer3(_make_layer(256, layers[2], 2)),
    layer4(_make_layer(512, layers[3], 2)),
    fc(512 * 4, num_classes)
{
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);
}

torch::Tensor ResNetImpl::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = torch::avg_pool2d(x, 7, 1);
    // torch::nn::AvgPool2d model(torch::nn::AvgPool2dOptions(7).stride(1));
    // x = model->forward(x);
    x = x.view({x.sizes()[0], -1});
    x = fc->forward(x);

    return x;
}

torch::nn::Sequential ResNetImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride) {
    torch::nn::Sequential downsample;
    // int expansion = BottleNeck::expansion;
    if (stride != 1 || inplanes != planes * expansion) {
        downsample = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes * expansion, 1)
                                .stride(stride).padding(0).bias(false)),
            torch::nn::BatchNorm2d(planes *  expansion)
        );
    }
    torch::nn::Sequential layers;
    layers->push_back(BottleNeck(inplanes, planes, stride, downsample));
    inplanes = planes * expansion;
    for (int64_t i = 1; i < blocks; i++) {
        layers->push_back(BottleNeck(inplanes, planes));
    }

    return layers;
}

ResNet resnet50(){
  ResNet model(std::vector<int>{3, 4, 6, 3});
  return model;
}