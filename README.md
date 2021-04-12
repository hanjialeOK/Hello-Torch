# Hello Torch

## LibTorch Installation

```c
// if only cpu
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.7.1%2Bcpu.zip
// if gpu with nvidia driver and cuda
wget https://download.pytorch.org/libtorch/cu110/libtorch-shared-with-deps-1.7.1%2Bcu110.zip
unzip libtorch-shared-with-deps-1.7.1+cu110.zip
export TORCH_DIR=/path/to/libtorch
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

if you want to download previous versions, click [here](https://blog.csdn.net/weixin_43742643/article/details/114156298).

## Compile & Run

```c
git clone https://github.com/hanjialeOK/Hello-Torch.git HelloTorch
cd HelloTorch
mkdir build && cd build
cmake ..
make
```

## Generate pre_trained weights

```c
import torch
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model = model.cuda()
model.eval()
// for k, v in model.named_parameters() : print(k)
var = torch.ones((1, 3, 224, 224))
var = var.cuda()
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("resnet50.pt")
```