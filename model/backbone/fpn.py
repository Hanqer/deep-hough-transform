import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None, groups=1,
                 base_width=64):
        super(BasicBlock, self).__init__()
        if BatchNorm is None:
            BatchNorm = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None, groups=1,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False, groups=groups)
        self.bn2 = BatchNorm(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, arch, block, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        self.layers = layers
        self.arch = arch
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]

        if arch == 'resnext50':
            self.base_width = 4
            self.groups = 32
        else:
            self.base_width = 64
            self.groups = 1
        
        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        if self.arch == 'resnet18':
             self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        else:
            self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        
        if self.arch == 'resnet18':
            self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
            self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
            self.latlayer3 = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)
        else: 
            self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
            self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
            self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm, groups=self.groups, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm, groups=self.groups, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, input):
        # Bottom-up
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)  # x4
        c3 = self.layer2(c2) #x8
        c4 = self.layer3(c3) #x16
        c5 = self.layer4(c4) #x16
        # Top-down
        p5 = self.toplayer(c5)
        p4 = nn.functional.upsample(p5, size=c4.size()[2:], mode='bilinear') + self.latlayer1(c4)
        p3 = nn.functional.upsample(p4, size=c3.size()[2:], mode='bilinear') + self.latlayer2(c3)
        p2 = nn.functional.upsample(p3, size=c2.size()[2:], mode='bilinear') + self.latlayer3(c2)
        
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        
        return p2, p3, p4, p5

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        if self.arch == 'resnet101':
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        elif self.arch == 'resnet50':
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        elif self.arch == 'resnet18':
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        elif self.arch == 'resnext50':
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def FPN101(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet('resnet101', Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

def FPN50(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet('resnet50', Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

def FPN18(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = ResNet('resnet18', BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, pretrained=pretrained)
    return model

def ResNext50_FPN(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = ResNet('resnext50', Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

if __name__ == "__main__":
    import torch
    model = FPN101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    input = torch.rand(1, 3, 480, 640)
    output = model(input)
    for out in output:
        print(out.size())
    # print(low_level_feat.size())

