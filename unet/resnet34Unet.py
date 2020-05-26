import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BasicBlock.expansion, kernel_size=3, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    
    def forward(self, x):

        return nn.ReLU(inplace=True)(self.residual_function(x)+self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_class=100):
        super().__init__()
        
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        for stride in strides:
            block(self.in_channels, out_channels, stride)
