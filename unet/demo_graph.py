import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter



class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

    def forward(self, x):
        return x * 2


model = SimpleModel()
dummy_input = (torch.zeros(1, 2, 3),)

with SummaryWriter(comment='./runs/constantModel') as w:
    w.add_graph(model, dummy_input, True)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


dummy_input = torch.rand(1, 3, 224, 224)

with SummaryWriter(comment='./runs/basicblock') as w:
    model = BasicBlock(3, 3)
    w.add_graph(model, (dummy_input, ), verbose=True)




# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         self.cnn1 = Net1()

#     def forward_once(self, x):
#         output = self.cnn1(x)
#         return output

#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
#         return output1, output2

# model = SiameseNetwork()
# with SummaryWriter(comment='./runs/SiameseNetwork') as w:
#     w.add_graph(model, (dummy_input, dummy_input))


# dummy_input = torch.Tensor(1, 3, 224, 224)

# with SummaryWriter(comment='./runs/alexnet') as w:
#     model = torchvision.models.alexnet()
#     w.add_graph(model, (dummy_input, ))

# with SummaryWriter(comment='./runs/vgg19') as w:
#     model = torchvision.models.vgg19()
#     w.add_graph(model, (dummy_input, ))

# with SummaryWriter(comment='./runs/densenet121') as w:
#     model = torchvision.models.densenet121()
#     w.add_graph(model, (dummy_input, ))

with SummaryWriter(comment='./runs/resnet18') as w:
    model = torchvision.models.resnet18()
    w.add_graph(model, (dummy_input, ))



