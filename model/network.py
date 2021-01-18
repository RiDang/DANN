import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.rvs import rvs
import model.resnet as resnet
import model.vgg as vgg
from .transformer_base.pos_Models import Encoder
pretrained_path = ['/home/dh/zdd/data/pretrained_model/vgg11-bbd30ac9.pth', \
    '/home/dh/zdd/data/pretrained_model/resnet18-5c106cde.pth', \
    '/home/dh/zdd/data/pretrained_model/resnet34-333f7ec4.pth',
    '/home/dh/zdd/data/pretrained_model/resnet50-19c8e357.pth']

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class extract_vgg(nn.Module):
    def __init__(self, args, pretrained=True):
        super(extract_vgg, self).__init__()
         
        self.args = args
        # self.net =  resnet.__dict__['resnet34']() 
        self.net =  vgg.__dict__['vgg11']() 
        if pretrained:
            pre_parameters = torch.load(pretrained_path[0])
            self.net.load_state_dict(pre_parameters)
            print('load model')
        self.features = self.net.features
        self.avgpool = self.net.avgpool
        self.classifier = self.net.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[:5](x)
        fts = x
        x = self.classifier[5:](x)
        return x, fts


class extract_resnet50(nn.Module):
    def __init__(self, args,  pretrained=True):
        super(extract_vgg, self).__init__()
         
        self.args = args
        # self.net =  resnet.__dict__['resnet34']() 
        self.net =  vgg.__dict__['resnet50']() 
        if pretrained:
            pre_parameters = torch.load(pretrained_path[2])
            self.net.load_state_dict(pre_parameters)
            print('load model')
        self.features = self.net.features
        self.classifier = self.net.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.classifier[:5](x)
        fts = x
        x = self.classifier[5:](x)
        return x, fts


class self_attn(nn.Module):
    def __init__(self, args,  pretrained=True):
        super(self_attn, self).__init__()
         
        self.args = args
        l_in = 4096
        self.classifier = nn.Sequential(
                   nn.ReLU(inplace=True),
                   nn.Dropout(),
                   nn.Linear(4096*12, 1024),  # 4096--81902,512--10240
                   nn.ReLU(inplace=True),
                   nn.Linear(1024, args.num_classes))
        
        self.encoder = Encoder( d_model=l_in, d_inner=2048,
                       len_max_seq = args.views, d_word_vec = l_in,
                       n_layers=args.nLayer, n_head=1, d_k=64, d_v=64, dropout=0.1)
        self.position = torch.Tensor(range(1, args.views+1)).repeat( args.batchsize,1).long().to(args.device)
       

    def forward(self, data):
        out,slf_attn = self.encoder(data, return_attns=True)
        out = out.reshape(out.shape[0], -1)
        # out = out.mean(dim=-2) # sum(dim=-2)

        features = self.classifier[0 : 3](out)
        out = self.classifier[3 :](features)
        return out, features
        # return out, nn.functional.normalize(features)


class res_attn(nn.Module):
    def __init__(self, args,  pretrained=True):
        super(res_attn, self).__init__()
         
        self.args = args
        

        self.net =  resnet.__dict__['resnet18']() 
        if pretrained:
            pre_parameters = torch.load(pretrained_path[1])
            self.net.load_state_dict(pre_parameters)
            print('load model')

        l_in = 512
        self.classifier = nn.Sequential(
                   nn.ReLU(inplace=True),
                   nn.Dropout(),
                   nn.Linear(512*12, 1024),  # 4096--81902,512--10240
                   nn.ReLU(inplace=True),
                   nn.Linear(1024, args.num_classes))
        
        self.encoder = Encoder( d_model=l_in, d_inner=2048,
                       len_max_seq = args.views, d_word_vec = l_in,
                       n_layers=args.nLayer, n_head=1, d_k=64, d_v=64, dropout=0.1)
        self.position = torch.Tensor(range(1, args.views+1)).repeat( args.batchsize,1).long().to(args.device)
       

    def forward(self, data):
        out = self.net(data)
        out = out.reshape(-1, self.args.views, out.shape[-1])
        out,slf_attn = self.encoder(out, return_attns=True)
        out = out.reshape(out.shape[0], -1)
        # out = out.mean(dim=-2) # sum(dim=-2)

        features = self.classifier[0 : 3](out)
        out = self.classifier[3 :](features)
        return out, features
        # return out, nn.functional.normalize(features)

