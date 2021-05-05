#include <iostream>

#include "poolnet.h"

/* ConvertLayer */
ConvertLayerImpl::ConvertLayerImpl() :
    convert0(_make_convertlayer())
{
    register_module("convert0", convert0);
}

std::vector<torch::Tensor> ConvertLayerImpl::forward(std::vector<torch::Tensor> x) {
    std::vector<torch::Tensor> resl;
    for(int i = 0; i < x.size(); i++) {
        resl.push_back(convert0[i]->as<torch::nn::Sequential>()->forward(x[i]));
    }
    return resl;
}

torch::nn::ModuleList ConvertLayerImpl::_make_convertlayer() {
    torch::nn::ModuleList list;
    for(int i = 0; i < 5; i++) {
        list->push_back(torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/inplanes[i], /*out_channels=*/planes[i], /*kernel_size=*/1)
                                .stride(1).padding(0).bias(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions(/*inplace=*/true))
            )
        );
    }
    return list;
}

/* DeepPoolLayer */
DeepPoolLayerImpl::DeepPoolLayerImpl(int64_t inplanes_, int64_t planes_, bool need_x2_, bool need_fuse_) :
    inplanes(inplanes_),
    planes(planes_),
    need_x2(need_x2_),
    need_fuse(need_fuse_),
    pools(_make_poollayer()),
    convs(_make_convlayer()),
    conv_sum(torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/planes, /*kernel_size=*/3)
                .stride(1).padding(1).bias(false))),
    conv_sum_c(torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/planes, /*out_channels=*/planes, /*kernel_size=*/3)
                .stride(1).padding(1).bias(false)))
{
    register_module("pools", pools);
    register_module("convs", convs);
    register_module("conv_sum", conv_sum);
    if(need_fuse) {
        register_module("conv_sum_c", conv_sum_c);
    }
}

torch::Tensor DeepPoolLayerImpl::forward(torch::Tensor x, torch::Tensor x2, torch::Tensor x3) {
    auto x_size = x.sizes();
    auto resl = x.clone();
    for(int i = 0; i < 3; i++) {
        auto y = convs[i]->as<torch::nn::Sequential>()->forward(pools[i]->as<torch::nn::Sequential>()->forward(x));
        resl = torch::add(resl, torch::nn::functional::interpolate(/*input=*/y, 
                torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({x_size[2], x_size[3]}))
                    .mode(torch::kBilinear).align_corners(true)));
    }
    resl = torch::relu(resl);
    if(need_x2) {
        resl = torch::nn::functional::interpolate(/*input=*/resl, 
                    torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({x2.size(2), x2.size(3)}))
                        .mode(torch::kBilinear).align_corners(true));
    }
    resl = conv_sum->forward(resl);
    if(need_fuse) {
        resl = conv_sum_c->forward(torch::add(torch::add(resl, x2), x3));
    }
    return resl;
}

torch::nn::ModuleList DeepPoolLayerImpl::_make_poollayer() {
    torch::nn::ModuleList list;
    for(int i = 0; i < 3; i++) {
        list->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(pool_size[i]).stride(pool_size[i])));
    }
    return list;
}

torch::nn::ModuleList DeepPoolLayerImpl::_make_convlayer() {
    torch::nn::ModuleList list;
    for(int i = 0; i < 3; i++) {
        list->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/inplanes, /*kernel_size=*/3)
                            .stride(1).padding(1).bias(false)));
    }
    return list;
}

/* ScoreLayer */
ScoreLayerImpl::ScoreLayerImpl(int64_t inplanes) :
    score(torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/1, /*kernel_size=*/1)
                .stride(1).padding(0).bias(true)))
{
    register_module("score", score);
}

torch::Tensor ScoreLayerImpl::forward(torch::Tensor x, c10::IntArrayRef x_size) {
    x = score->forward(x);
    if(!x_size.empty()) {
        x = torch::nn::functional::interpolate(/*input=*/x, 
                torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({x_size[2], x_size[3]}))
                    .mode(torch::kBilinear).align_corners(true));
    }
    return x;
}

// void _make_extra_layer() {
//     auto convert_layers = ConvertLayer();

    // int64_t inplanes[5] = {};
    // int64_t planes[5] = {};
    // bool need_x2[5] = {};
    // bool need_fuse[5] = {};

    // for(int i = 0; i < 5; i++) {

    // }
// }

/* PoolNet */
PoolNetImpl::PoolNetImpl() :
    base(resnet50()),
    deep_pool(_make_deeppool_layers()),
    score(ScoreLayer()),
    convert(ConvertLayer())
{
    register_module("base", base);
    register_module("deep_pool", deep_pool);
    register_module("score", score);
    register_module("convert", convert);
}

torch::Tensor PoolNetImpl::forward(torch::Tensor x) {
    return x;
}

torch::nn::ModuleList PoolNetImpl::_make_deeppool_layers() {
    const int64_t inplanes[5] = { 512, 512, 256, 256, 128 };
    const int64_t planes[5] = { 512, 256, 256, 128, 128 };
    const bool need_x2[5] = { False, True, True, True, False };
    const bool need_fuse[5] = { True, True, True, True, False };
    torch::nn::ModuleList list;
    for(int i = 0; i < 5; i++) {
        list->push_back(DeepPoolLayer(inplanes[i], planes[i], need_x2[i], need_fuse[i]));
    }
    return list;
}