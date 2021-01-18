import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.rvs import rvs
from model.vgg import vgg11

pretrained_path = ['/home/dh/zdd/data/pretrained_model/wide_resnet50_2-95faca4d.pth', \
    '/home/dh/zdd/data/pretrained_model/resnet18-5c106cde.pth', \
    '/home/dh/zdd/data/pretrained_model/vgg11-bbd30ac9.pth']

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class myvgg(nn.Module):
    def __init__(self, nclasses=34, pretrained=True):
        super(myvgg, self).__init__()
        
        self.pretrained = pretrained
        self.net =  vgg11()
        #nn.init.normal_(self.fc1.weight, mean=0, std=np.sqrt(2.0 / (512*2)))


        if self.pretrained:
            print('load model')
            state_dict = torch.load(pretrained_path[2])
            self.net.load_state_dict(state_dict)

        self.net.classifier[-1]= nn.Linear(4096, nclasses) 
    def forward(self, x):
        x = self.net.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.classifier[:3](x)
        fts = x
        x = self.net.classifier[3:](x)
        return x,fts,fts
        #return x,fts,self.fc3
