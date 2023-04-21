import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import exp, pi
import numpy as np
import cv2 as cv
import scipy.stats as st
import matplotlib.pyplot as plt

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen))
    # out_filter = np.repeat(out_filter, channels, axis = 0)
    return out_filter   # kernel_size=21

class SeparableConv2d(nn.Module):
    def __init__(self):
        super(SeparableConv2d, self).__init__()
        kernel = gauss_kernel(21, 3, 3)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        ## kernel_point = [[1.0]]
        ## kernel_point = torch.FloatTensor(kernel_point).unsqueeze(0).unsqueeze(0)
        # kernel = torch.FloatTensor(kernel).expand(3, 3, 21, 21)   # torch.expand(）向输入的维度前面进行扩充，输入为三通道时，将weight扩展为[3,3,21,21]
        ## kernel_point = torch.FloatTensor(kernel_point).expand(3,3,1,1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        # self.pointwise = nn.Conv2d(1, 1, 1, 1, 0, 1, 1,bias=False)    # 单通道时in_channels=1，out_channels=1,三通道时，in_channels=3, out_channels=3  卷积核为随机的
        ## self.weight_point = nn.Parameter(data=kernel_point, requires_grad=False)

    # 输入两幅图像,进行高斯模糊
    # 将生成模糊图进行MSE损失计算
    def forward(self, img1,img2):
        img1 = F.conv2d(img1, self.weight, groups=1,padding=10)
        img2 = F.conv2d(img2, self.weight, groups=1,padding=10)
        ## x = F.conv2d(x, self.weight_point, groups=1, padding=0)  #卷积核为[1]
        # x = self.pointwise(x)
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        MSE_loss = loss_fn(img1,img2)
        return MSE_loss


