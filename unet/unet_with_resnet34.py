import numpy as np

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms

class UnetWithResNet34(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None, upsampling_method="conv_transpose"):
        super().__init__()
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)



if __name__ == "__main__":
    resnet34 = models.resnet34(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet34.parameters(), lr=0.001, momentum=0.9)