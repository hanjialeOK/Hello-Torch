import torch
import cv2
from networks.deeplab_resnet import Bottleneck, ResNet

def main():
    net = ResNet(Bottleneck, [3, 4, 6, 3])
    net = net.cuda()
    net.load_state_dict(torch.load("../models/resnet50_caffe.pth"), strict=False)
    # print(net)
    # net.eval()
    for k, v in net.named_parameters() : 
        print("%s %d" %(k, v.requires_grad))
    x = torch.ones(1, 3, 224, 224)
    x = x.cuda()
    out = net.forward(x)
    # traced_script_module = torch.jit.trace(net, x)
    # traced_script_module.save("../models/resnet50_caffe.pt")
    print(out[0][2000])
    print(out.size())

if __name__ == '__main__':
    main()