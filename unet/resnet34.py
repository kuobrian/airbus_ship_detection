import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

class decoderblock(nn.Module):
    def __inin__(self, in_channel, num_filter, kernel_size, is_deconv=False):
        super().__init__()
        print("asd")



if __name__ == "__main__":
    resnet = models.resnet34(pretrained=True)
    # print(resnet)
    print(resnet.layer4)
    print(len(resnet.layer4))
