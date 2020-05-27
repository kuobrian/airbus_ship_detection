import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchsummary import summary
from tensorboardX import SummaryWriter




def double_conv(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

def resnet34Unet(in_channel, out_channel):
    return ResNet(in_channel, out_channel, BasicBlock, [3, 4, 6, 3])



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
                nn.Conv2d(in_channels, out_channels*BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    
    def forward(self, x):
        a = self.residual_function(x)
        b = self.shortcut(x)
        return nn.ReLU(inplace=True)(a+b)

class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel, block, num_block):
        super().__init__()
        
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size = 7, stride = 2, padding = 3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_down = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_down = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_down = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_down = self._make_layer(block, 512, num_block[3], 2)

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, out_channel)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_up = double_conv(256 + 512, 256)
        self.conv2_up = double_conv(128 + 256, 128)
        self.conv3_up = double_conv(64 + 128, 64)
        
        self.dconv_last = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_channel, 1)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out_conv1= self.conv1(x)
        temp = self.pool1(out_conv1)
        out_conv2 = self.conv2_down(temp)
        out_conv3 = self.conv3_down(out_conv2)
        out_conv4 = self.conv4_down(out_conv3)
        bottle = self.conv5_down(out_conv4)

        # output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)
        up_x = self.upsample(bottle)
        up_x = torch.cat([up_x, out_conv4], dim=1)
        up_x = self.conv1_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv3], dim=1)
        up_x = self.conv2_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv2], dim=1)
        up_x = self.conv3_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv1], dim=1)
        out = self.dconv_last(up_x)

        return out

class Resnet34Unet(nn.Module):
    def __init__(self, in_channel, out_channel, training=True):
        super().__init__()
        self.training = training
        resnet34 = models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(*list(resnet34.children())[:3])
        self.pool1 = resnet34.maxpool
        self.conv2_down = nn.Sequential(*list(resnet34.children())[4])
        self.conv3_down = nn.Sequential(*list(resnet34.children())[5])
        self.conv4_down = nn.Sequential(*list(resnet34.children())[6])
        self.conv5_down = nn.Sequential(*list(resnet34.children())[7])
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_up = double_conv(256 + 512, 256)
        self.conv2_up = double_conv(128 + 256, 128)
        self.conv3_up = double_conv(64 + 128, 64)

        self.dconv_last = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_channel, 1)
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        temp = self.pool1(out_conv1)
        out_conv2 = self.conv2_down(temp)
        out_conv3 = self.conv3_down(out_conv2)
        out_conv4 = self.conv4_down(out_conv3)
        bottle = self.conv5_down(out_conv4)

        up_x = self.upsample(bottle)
        up_x = torch.cat([up_x, out_conv4], dim=1)
        up_x = self.conv1_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv3], dim=1)
        up_x = self.conv2_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv2], dim=1)
        up_x = self.conv3_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv1], dim=1)
        out = self.dconv_last(up_x)

        return out





# if __name__ == "__main__":
#     import time

#     start = time.time()
#     # x = torch.randn(5, 3, 512, 512)
#     # model = resnet34Unet(3, 3)
#     # out = model(x)
#     # print(out.shape, time.time()-start)
    
#     # Using pre-trained model
#     model = Resnet34Unet(3, 3)

#     # Frozen parameters of the encoder layers    
#     for child in model.children():
#         child_counter +=1
#         for param in child.parameters():
#             param.requires_grad = False
#         if child_counter == 6:
#             break
            


    # if torch.cuda.is_available():
    #     model.cuda()
    
    
    
    
    
    
    
    
    
    # with SummaryWriter(comment='./runs/Resnet34Unet') as w:
    #     w.add_graph(m, (x, ), verbose=True)
    # summary(m, (3, 512, 512))
    




