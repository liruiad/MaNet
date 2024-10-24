import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms 

class MaAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(MaAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inp)
        self.act1 = nn.ReLU()
    
    def mu(self, x):
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3)) + 0.000000001) / (
                    x.shape[2] * x.shape[3]))
    def forward(self, x):
        identity = x
        b,c,h,w = x.size()

        y= torch.pow(self.sigma(x),2)
        y= y.sigmoid()
        y=y.view(b,c,1,1)
        x= x*y.expand_as(x)
        x = self.bn1(x)
             
        x_h = self.pool_h(x) # b,c,h,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # b,c,w,1

        x_h = self.conv1(x_h)
        x_h = self.act1(x_h)

        x_w = self.conv1(x_w)
        x_w = self.act1(x_w)

        y = torch.cat([x_h, x_w], dim=2) # b,c,h+w,1
        y = self.conv2(y) # b, mip, h+w, 1
        
        x_h, x_w = torch.split(y, [h, w], dim=2) # b, mip, h, 1
        x_w = x_w.permute(0, 1, 3, 2) # b, mip, 1, w

        a_h = x_h.sigmoid() # b, c, h, 1
        a_w = x_w.sigmoid() # b, c, 1, w

        out = identity * a_w * a_h

        return out

    
