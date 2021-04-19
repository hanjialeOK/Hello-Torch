#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "networks/deeplab_resnet.h"

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
    std::cout << "loading weights..." << std::endl;
    torch::load(net, "../models/resnet50.pt");
    std::cout << "weights have been loaded!" << std::endl;

    for (const auto& p : net->parameters()) {
        // std::cout << p << std::endl;
        // p.requires_grad() = false;
        // p.set_requires_grad(false);
        // p.requires_grad_(false);
    }
    for (const auto& pair : net->named_parameters()) {
        // std::cout << pair.key() << ": " << pair.value() << std::endl;
        std::cout << pair.key() << " " << pair.value().requires_grad() << std::endl;
    }

    int input_size = 224;
    cv::Mat img;

    img = cv::imread("../images/1.jpg");

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(input_size, input_size));

    // auto img_tensor = torch::from_blob(img.data, {1, img.cols, img.rows, 3}, torch::kByte).to(device);
    auto img_tensor = torch::from_blob(img.data, {1, input_size, input_size, 3}, torch::kByte).to(device);
    img_tensor = img_tensor.permute({0,3,1,2});
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.div(255.0);

    // std::cout << net->forward(img_tensor) << std::endl;

    return 0;
}