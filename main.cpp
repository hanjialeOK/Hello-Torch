#include <stdio.h>
#include <string.h>
#include <dirent.h>

#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>

#include "networks/poolnet.h"

int main() {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }

    /* net */
    torch::jit::script::Module net = torch::jit::load("../models/poolnet.pt");
    net.to(device);
    // torch::NoGradGuard no_grad;
    net.eval();

    int input_size = 224;
    cv::Mat img;

	DIR *dir = NULL;
	struct dirent *file;
	if((dir = opendir("../images/")) == NULL) {  
		printf("opendir failed!");
		return -1;
	}

	int cnt = 0;
    while(file = readdir(dir)) {
		// 判断是否为文件
		if (file->d_type != 8) continue;

        char name[20] = "\0", type[5] = "\0";
        strncpy(name, file->d_name, 11*sizeof(char));
        name[11] = 0;
        strncpy(type, file->d_name + 12, 3*sizeof(char));
        type[3] = 0;

        if (strcmp(type, "jpg") != 0) continue;

		std::cout << file->d_name << std::endl;
		// 为文件加上相对路径
		char fileName[30] = "../images/";
		strcat(fileName, file->d_name);

        img = cv::imread(fileName);

        // char who[10] = "\0",newname[20] = "../test/";
        // sprintf(who, "%2d.jpg", cnt++);
        // if(who[0] == ' ') who[0] = '0';
        // strcat(newname, who);
        // cv::imwrite(newname, img);
        // continue;

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        // cv::resize(img, img, cv::Size(input_size, input_size));

        auto img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte).to(device);
        img_tensor = img_tensor.permute({0,3,1,2});
        img_tensor = img_tensor.toType(torch::kFloat);
        // img_tensor = img_tensor.div(255.0);

        auto start = std::chrono::high_resolution_clock::now();

        torch::NoGradGuard no_grad;
        auto out = net.forward({img_tensor}).toTensor();

        std::cout << out.type() << " " << out.sizes() << " " << out.requires_grad() << std::endl;

        // out.squeeze_(0);
        out.squeeze_();
        out.sigmoid_();
        out.mul_(255.0);
        out = out.toType(torch::kByte);
        // out = out.permute({1, 2, 0});
        // out = out.expand({out.size(0), out.size(1), 3});
        // std::cout << "is_contiguous(out): " << out.is_contiguous() << std::endl;
        // out = out.contiguous();
        out = out.detach();

        auto start1 = std::chrono::high_resolution_clock::now();
        out = out.to(torch::kCPU);
        std::cout << out.type() << " " << out.sizes() << " " << out.requires_grad() << std::endl;
        auto end1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
        // It should be known that it takes longer time at first time
        std::cout << "GPU to CPU " << duration1.count() << " ms" << std::endl;

        // out = out.squeeze().sigmoid().mul(255.0).toType(torch::kByte).to(torch::kCPU);

        // if (strcmp(name, "2007_004000") == 0) {
        //     std::cout << out << std::endl;
        // }

        // auto out_data = out.packed_accessor32<unsigned char, 2>();
        // for(int i = 0; i < out_data.size(0); i++) {
        //     for(int j = 0; j < out_data.size(1); j++) {
        //         int x = out_data[i][j];
        //     }
        // }

        cv::Mat pred;
        pred.create(cv::Size(img.cols, img.rows), CV_8UC1);
        memcpy(pred.data, out.data_ptr(), out.numel() * sizeof(torch::kByte));
        // cudaMemcpy(pred.data, out.data_ptr(), out.numel(), cudaMemcpyDeviceToHost);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // It should be known that it takes longer time at first time
        std::cout << "inference taken : " << duration.count() << " ms" << std::endl;

        char saveName[30] = "../images/";
		strcat(saveName, name);
        strcat(saveName, ".png");
        cv::imwrite(saveName, pred);

	}
    closedir(dir);
    return 0;
}