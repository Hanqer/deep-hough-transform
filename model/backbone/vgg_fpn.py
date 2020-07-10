import torch
import math
from torch import nn

# vgg16
vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

class VGG_Feature(nn.Module):
    def __init__(self, extract=[8, 15, 22, 29]):
        super(VGG_Feature, self).__init__()
        self.vgg = nn.ModuleList(vgg(cfg=vgg_config))
        self.extract = extract

    def forward(self, x):
        features = []
        for i in range(len(self.vgg)):
            x = self.vgg[i](x)
            if i in self.extract:
                features.append(x)
        return features

class VGG_FPN(nn.Module):
    def __init__(self):
        super(VGG_FPN, self).__init__()
        self.vgg_feat = VGG_Feature()

        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self._init_weight()
        self._load_pretrained_model()
    
    def forward(self, x):
        [c2, c3, c4, c5] = self.vgg_feat(x)
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
        self.vgg_feat.vgg.load_state_dict(torch.load('/home/hanqi/vgg16_feat.pth'))