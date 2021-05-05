#include<torch/script.h>
#include<torch/torch.h>

#include "deeplab_resnet.h"

class ConvertLayerImpl : public torch::nn::Module {
public:
    ConvertLayerImpl();
    torch::Tensor forward(std::vector<torch::Tensor> x);
    torch::nn::ModuleList _make_convertlayer();
private:
    int64_t inplanes[5] = { 64, 256, 512, 1024, 2048 };
    int64_t planes[5] = { 128, 256, 256, 512, 512 };
    torch::nn::ModuleList convert0;
};
TORCH_MODULE(ConvertLayer);

class DeepPoolLayerImpl : public torch::nn::Module {
public:
    DeepPoolLayerImpl(int64_t inplanes_, int64_t planes_, bool need_x2_, bool need_fuse_);
    torch::Tensor forward(torch::Tensor x, torch::Tensor x2 = NULL, torch::Tensor x3 = NULL);
    torch::nn::ModuleList _make_poollayer();
    torch::nn::ModuleList _make_convlayer();
private:
    int64_t inplanes, planes;
    bool need_x2, need_fuse;
    int64_t pool_size =  { 2, 4, 8 };
    torch::nn::ModuleList pools, convs;
    torch::nn::Conv2d conv_sum, conv_sum_c;
};
TORCH_MODULE(DeepPoolLayer);