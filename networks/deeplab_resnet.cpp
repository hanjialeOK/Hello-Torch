#include <iostream>

#include "deeplab_resnet.h"

/* BottleNeck */

BottleNeckImpl::BottleNeckImpl(int64_t inplanes, int64_t planes, int64_t stride, int64_t dilation,
    torch::nn::Sequential downsample_) : 
    downsample(downsample_),
    conv1(torch::nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/planes, /*kernal_size=*/1)
            .stride(stride).padding(0).bias(false)),
    bn1(torch::nn::BatchNorm2dOptions(planes).affine(true)),
    conv2(torch::nn::Conv2dOptions(/*in_channels=*/planes, /*out_channels=*/planes, /*kernal_size=*/3)
            .stride(1).padding(dilation).bias(false).dilation(dilation)),
    bn2(torch::nn::BatchNorm2dOptions(planes).affine(true)),
    conv3(torch::nn::Conv2dOptions(/*in_channels=*/planes, /*out_channels=*/planes * expansion, /*kernal_size=*/1)
            .stride(1).padding(0).bias(false)),
    bn3(torch::nn::BatchNorm2dOptions(planes * expansion).affine(true))
{
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    for(const auto& p : bn1->parameters()) {
        p.requires_grad_(false);
    }
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    for(const auto& p : bn2->parameters()) {
        p.requires_grad_(false);
    }
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    for(const auto& p : bn3->parameters()) {
        p.requires_grad_(false);
    }

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
ResNetImpl::ResNetImpl(std::vector<int> layers) : 
    conv1(torch::nn::Conv2dOptions(/*in_channels=*/3, /*out_channels=*/64, /*kernal_size=*/7)
            .stride(2).padding(3).bias(false)),
    bn1(torch::nn::BatchNorm2dOptions(64).affine(true)),
    layer1(_make_layer(/*planes=*/64,  /*blocks=*/layers[0])),
    layer2(_make_layer(/*planes=*/128, /*blocks=*/layers[1], /*stride=*/2)),
    layer3(_make_layer(/*planes=*/256, /*blocks=*/layers[2], /*stride=*/2)),
    layer4(_make_layer(/*planes=*/512, /*blocks=*/layers[3], /*stride=*/1, /*dilation=*/2))
{
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    for(const auto& p : bn1->parameters()) {
        p.requires_grad_(false);
    }
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);

    /**
     * 1. actually, this method is to initilize weights, which is unuseful if you train with a pre_trained weight.
     * 2. "&as<>()" could convert nn::module to nn::Conv2d, I got this funtion from module.h.
     **/
    // for(const auto& m : this->modules(/*include_self=*/false)) {
    //     if(m->as<torch::nn::Conv2d>() != nullptr) {
    //         /* a leaf Variable that requires grad is being used in an in-place operation. */
    //         // std::cout << m->name() << " " << m->parameters().size() << std::endl;
    //         // m->parameters()[0].normal_(0, 0.01);
    //     }
    //     else if(m->as<torch::nn::BatchNorm2d>() != nullptr) {
    //         // std::cout << m->name() << " " << m->parameters().size() << std::endl;
    //         m->parameters()[0].fill_(1); /* weight */
    //         m->parameters()[1].zero_(); /* bias */
    //     }
    // }
    for(const auto& m : this->modules(/*include_self=*/false)) {
        if(auto* conv = m->as<torch::nn::Conv2d>()) {
            /* a leaf Variable that requires grad is being used in an in-place operation. */
            // conv->weight.normal_(0, 0.01);
        }
        else if(auto* bn = m->as<torch::nn::BatchNorm2d>()) {
            bn->weight.fill_(1);
            bn->bias.zero_();
        }
    }
}

std::vector<torch::Tensor> ResNetImpl::forward(torch::Tensor x) {
    std::vector<torch::Tensor> tmp_x;
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    tmp_x.push_back(x);
    x = torch::max_pool2d(/*tensor=*/x, /*kernel_size=*/3, /*stride=*/2, /*padding=*/1, /*dilation=*/1, /*ceil_mode=*/true);
    torch::nn::MaxPool2d pool(torch::nn::MaxPool2dOptions(3).stride(2).padding(1).ceil_mode(true));
    x = pool->forward(x);

    x = layer1->forward(x);
    tmp_x.push_back(x);
    x = layer2->forward(x);
    tmp_x.push_back(x);
    x = layer3->forward(x);
    tmp_x.push_back(x);
    x = layer4->forward(x);
    tmp_x.push_back(x);

    return tmp_x;
}

torch::nn::Sequential ResNetImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride, int64_t dilation) {
    torch::nn::Sequential downsample;
    if (stride != 1 || inplanes != planes * expansion || dilation == 2 || dilation == 4) {
        downsample = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/planes * expansion, /*kernel_size=*/1)
                                .stride(stride).padding(0).bias(false)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes * expansion).affine(true))
        );
    }
    for(const auto& p : downsample->ptr(1)->parameters()) {
        p.requires_grad_(false);
    }
    torch::nn::Sequential layers;
    layers->push_back(BottleNeck(inplanes, planes, stride, dilation, downsample));
    inplanes = planes * expansion;
    for (int64_t i = 1; i < blocks; i++) {
        layers->push_back(BottleNeck(inplanes, planes, /*stride=*/1, dilation));
    }

    return layers;
}

/* ResNet_locate */
ResNet_locateImpl::ResNet_locateImpl(std::vector<int> layers) : 
    resnet(layers),
    ppms_pre(torch::nn::Conv2dOptions(/*in_channels=*/2048, /*out_channels=*/512, /*kernal_size=*/1)
            .stride(1).padding(0).bias(false)),
    ppms(_make_modulelist_ppms()),
    ppms_cat(torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/2048, /*out_channels=*/512, /*kernel_size=*/3)
                                .stride(1).padding(1).bias(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions(/*inplace=*/true))
            )
        ),
    infos(_make_modulelist_infos())
{
    register_module("resnet", resnet);
    register_module("ppms_pre", ppms_pre);
    register_module("ppms", ppms);
    register_module("ppms_cat", ppms_cat);
    register_module("infos", infos);
}

torch::nn::ModuleList ResNet_locateImpl::_make_modulelist_ppms() {
    torch::nn::ModuleList list;
    for(int i = 1; i <= 5; i += 2) {
        list->push_back(torch::nn::Sequential(
            torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(/*output_size=*/i)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/inplanes, /*kernel_size=*/1)
                                .stride(1).padding(0).bias(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions(/*inplace=*/true))
            )
        );
    }
    return list;
}

torch::nn::ModuleList ResNet_locateImpl::_make_modulelist_infos() {
    torch::nn::ModuleList list;
    for(int i = 0; i < 4; i++) {
        list->push_back(torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/planes[i], /*kernel_size=*/3)
                                .stride(1).padding(1).bias(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions(/*inplace=*/true))
            )
        );
    }
    return list;
}

torch::Tensor ResNet_locateImpl::forward(torch::Tensor x) {
    auto tmp_x = resnet->forward(x);
    auto y = ppms_pre->forward(tmp_x.back());
    return y;
}

ResNet_locate resnet50() {
    ResNet_locate model(std::vector<int>{1, 1, 1, 1});
    return model;
}