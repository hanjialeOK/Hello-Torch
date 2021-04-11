#include <torch/torch.h>
#include <iostream>

#include "networks/ResNet.h"

struct NetImpl : torch::nn::Module {
    NetImpl(int k) : 
        conv1(torch::nn::Conv2dOptions(k, 256, 4)
                .bias(false)),
        batch_norm1(256),
        conv2(torch::nn::Conv2dOptions(256, 128, 3)
                .stride(2)
                .padding(1)
                .bias(false)),
        batch_norm2(128),
        conv3(torch::nn::Conv2dOptions(128, 64, 4)
                .stride(2)
                .padding(1)
                .bias(false)),
        batch_norm3(64),
        conv4(torch::nn::Conv2dOptions(64, 1, 4)
                .stride(2)
                .padding(1)
                .bias(false))
    {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::tanh(conv4(x));
        return x;
    }

    torch::nn::Conv2d conv1, conv2, conv3, conv4;
    torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(Net);

int main() {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }
    // Net net(100);
    ResNet net = resnet50();
    net->to(device);
    for (const auto& pair : net->named_parameters()) {
        // std::cout << pair.key() << ": " << pair.value() << std::endl;
        std::cout << pair.key() << std::endl;
    }
    torch::Tensor input = torch::randn({1, 3, 224, 224}, device);
    std::cout << net->forward(input) << std::endl;
}