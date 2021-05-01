#include <torch/torch.h>
#include <iostream>

struct NetImpl : torch::nn::Module {
    NetImpl(int k) : 
        conv1(torch::nn::Conv2dOptions(k, 256, 1)
                .bias(false)),
        batch_norm1(256),
        conv2(torch::nn::Conv2dOptions(256, 128, 1)
                .stride(2)
                .padding(1)
                .bias(false)),
        batch_norm2(128),
        conv3(torch::nn::Conv2dOptions(128, 64, 1)
                .stride(2)
                .padding(1)
                .bias(false)),
        batch_norm3(64),
        conv4(torch::nn::Conv2dOptions(64, 1, 1)
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
    Net net(3);
    net->to(device);
    net->eval();
    auto x = torch::ones({1, 3, 224, 224}, torch::kFloat).to(device);
    torch::NoGradGuard no_grad;
    auto out = net->forward(x);
    std::cout << out << std::endl;
}