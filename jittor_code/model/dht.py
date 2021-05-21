import jittor as jt
from jittor import Function, nn
import numpy as np 
import matplotlib.pyplot as plt 

from model.cuda_src import cuda_src_forward as csf
from model.cuda_src import cuda_src_backward as csb


class DHT_Func(Function):    
    def execute(self, x, numangle, numrho):
        n, c, h, w = x.shape
        cuda_src_forward = csf.replace('#numangle', str(numangle))
        cuda_src_forward = cuda_src_forward.replace('#numrho', str(numrho))

        irho = int((h*h + w*w)**0.5 + 1) / float((numrho - 1))
        itheta = 3.14159265358979323846 / numangle
        angle = jt.arange(numangle) * itheta
        tabCos = angle.cos() / irho
        tabSin = angle.sin() / irho 

        output = jt.code([n, c, numangle, numrho], x.dtype, [x, tabCos, tabSin],
        cuda_src=cuda_src_forward)
        
        self.save_vars = x, numangle, numrho
        return output

    def grad(self, grad):
        x, numangle, numrho = self.save_vars
        cuda_src_backward = csb.replace('#numangle', str(numangle))
        cuda_src_backward = cuda_src_backward.replace('#numrho', str(numrho))

        irho = int((h*h + w*w)**0.5 + 1) / float((numrho - 1))
        itheta = 3.14159265358979323846 / numangle
        angle = jt.arange(numangle) * itheta
        tabCos = angle.cos() / irho
        tabSin = angle.sin() / irho 

        return jt.code([x.shape], [x.dtype], [x, grad, tabCos, tabSin],
        cuda_src=cuda_src_backward)

class C_dht(nn.Module):
    def __init__(self, numAngle, numRho):
        super(C_dht, self).__init__()
        self.numAngle = numAngle
        self.numRho = numRho
    
    def execute(self, feat):
      return DHT_Func.apply(feat, self.numAngle, self.numRho)


class DHT(nn.Module):
    def __init__(self, numAngle, numRho):
        super(DHT, self).__init__()       
        self.line_agg = C_dht(numAngle, numRho)

    def execute(self, x):
        accum = self.line_agg(x)
        return accum


class DHT_Layer(nn.Module):
    def __init__(self, input_dim, dim, numAngle, numRho):
        super(DHT_Layer, self).__init__()
        self.fist_conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.dht = DHT(numAngle=numAngle, numRho=numRho)
        self.convs = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
    def execute(self, x):
        x = self.fist_conv(x)
        x = self.dht(x)
        x = self.convs(x)
        return x

