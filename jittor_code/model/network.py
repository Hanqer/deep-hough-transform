import jittor as jt
import numpy as np 
from jittor import nn

from model.backbone.fpn import FPN50
from model.dht import DHT_Layer


class Net(nn.Module):
    def __init__(self, numAngle, numRho, backbone):
        super(Net, self).__init__()
        if backbone == 'resnet50':
            print('using resnet50 backbone.')
            self.backbone = FPN50(pretrained=True, output_stride=16)
            output_stride = 16
        else:
            raise NotImplementedError
        

        self.dht_detector1 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho)
        self.dht_detector2 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho // 2)
        self.dht_detector3 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho // 4)
        self.dht_detector4 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho // (output_stride // 4))
            
        self.last_conv = nn.Sequential(
            nn.Conv2d(512, 1, 1)
        )

        self.numAngle = numAngle
        self.numRho = numRho

    def upsample_cat(self, p1, p2, p3, p4):
        p1 = nn.interpolate(p1, size=(self.numAngle, self.numRho), mode='bilinear', align_corners=True)
        p2 = nn.interpolate(p2, size=(self.numAngle, self.numRho), mode='bilinear', align_corners=True)
        p3 = nn.interpolate(p3, size=(self.numAngle, self.numRho), mode='bilinear', align_corners=True)
        p4 = nn.interpolate(p4, size=(self.numAngle, self.numRho), mode='bilinear', align_corners=True)
        return jt.concat([p1, p2, p3, p4], dim=1)

    def execute(self, x):
        p1, p2, p3, p4 = self.backbone(x)

        p4 = self.dht_detector4(p4)
        p3 = self.dht_detector3(p3)
        p2 = self.dht_detector2(p2)
        p1 = self.dht_detector1(p1)
        
        cat = self.upsample_cat(p1, p2, p3, p4)

        logist = self.last_conv(cat)

        return logist
