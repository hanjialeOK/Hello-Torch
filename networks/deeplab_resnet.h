#include<torch/script.h>
#include<torch/torch.h>

class BottleNeckImpl : public torch::nn::Module {
public:
    BottleNeckImpl(int64_t inplanes, int64_t planes, int64_t stride_ = 1, int64_t dilation_ = 1,
        torch::nn::Sequential downsample_ = torch::nn::Sequential());
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Sequential downsample;
    torch::nn::Conv2d conv1,conv2, conv3;
    torch::nn::BatchNorm2d bn1, bn2, bn3;
};
TORCH_MODULE(BottleNeck);

const int64_t expansion = 4;

// template <class Block>
class ResNetImpl : public torch::nn::Module {
public:
    ResNetImpl(std::vector<int> layers);
    std::vector<torch::Tensor> forward(torch::Tensor x);
    // std::vector<torch::Tensor> features(torch::Tensor x, int encoder_depth = 5);
    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1, int64_t dilation = 1);
	// std::vector<torch::nn::Sequential> get_stages();
	// void make_dilated(std::vector<int> stage_list, std::vector<int> dilation_list);
private:
    // const int64_t expansion = 1;
	int64_t inplanes = 64;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Sequential layer1, layer2, layer3, layer4;
};
TORCH_MODULE(ResNet);

class ResNet_locateImpl : public torch::nn::Module {
public:
    ResNet_locateImpl(std::vector<int> layers);
    torch::Tensor forward(torch::Tensor x);
    torch::nn::ModuleList _make_modulelist_ppms();
    torch::nn::ModuleList _make_modulelist_infos();
private:
    int64_t inplanes = 512;
    int64_t planes[4] = { 512, 256, 256, 128 };
    ResNet resnet;
    torch::nn::Conv2d ppms_pre;
    torch::nn::ModuleList ppms;
    torch::nn::Sequential ppms_cat;
    torch::nn::ModuleList infos;
};
TORCH_MODULE(ResNet_locate);

ResNet_locate resnet50();