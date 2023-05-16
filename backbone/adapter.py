import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class Istr_2(nn.Module):
    def __init__(self, block, num_classes=120):
        super(Istr_2, self).__init__()
        self.attention2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Sigmoid()
        )

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc2 = nn.Linear(512 * block.expansion, num_classes)

    def forward(self,x):
        #print(x.size())
        fea2 = self.attention2(x)
        fea2 = fea2 * x
        out2_feature = self.scala2(fea2).view(x.size(0), -1)
        #print(out2_feature.size())
        out2 = self.fc2(out2_feature)

        return out2

class Istr_1(nn.Module):
    def __init__(self, block, num_classes=100):
        super(Istr_1, self).__init__()
        self.attention1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=64 * block.expansion
            ),
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

    def forward(self,x):
        fea1 = self.attention1(x)
        fea1 = fea1 * x
        out1_feature = self.scala1(fea1).view(x.size(0), -1)
        out1 = self.fc1(out1_feature)

        return out1

class Istr_3(nn.Module):
    def __init__(self, block, num_classes=100):
        super(Istr_3, self).__init__()
        self.attention3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=256 * block.expansion
            ),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.fc3 = nn.Linear(512 * block.expansion, num_classes)

    def forward(self,x):
        fea3 = self.attention3(x)
        fea3 = fea3 * x
        out3_feature = self.scala3(fea3).view(x.size(0), -1)
        out3 = self.fc3(out3_feature)

        return out3
    
    
def adapter_1(num_classes=100):
    return Istr_1(BasicBlock,num_classes)

def adapter_2(num_classes=100):
    return Istr_2(Bottleneck,num_classes)

def adapter_3(num_classes=100):
    return Istr_3(BasicBlock,num_classes)
