import torch
import torch.nn as nn
import os
import sys
import torchvision
import torchvision.models as models
from torch.autograd import Variable
from .GCNet import GlobalContextBlock



# VGG+DECONVOLUTION+edgenetwork+FEATURCONSTRUCTION
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False
                              )  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Conv2dUnit(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(Conv2dUnit, self).__init__()

        self.conv = nn.ConvTranspose2d(in_planes, out_planes,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class BasicTransConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicTransConv2d, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_planes, out_planes,
                                            kernel_size=kernel_size, stride=stride,
                                            padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.transconv(x)))
        return x

class featureExtractionB(nn.Module):

    def __init__(self, in_planes):
        super(featureExtractionB, self).__init__()
        self.path1 = nn.Sequential(
            BasicConv2d(in_planes, 128, kernel_size=1, stride=1, padding=0),
            BasicConv2d(128, 192, kernel_size=7, stride=1, padding=3)
        )

        self.path2 = nn.Sequential(
            BasicConv2d(in_planes, 128, kernel_size=1, stride=1, padding=0),
            BasicConv2d(128, 192, kernel_size=3, stride=1, padding=1)
        )

        self.path3 = nn.Sequential(
            BasicConv2d(in_planes, 128, kernel_size=1, stride=1, padding=0),
            BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

        self.path4 = nn.Sequential(
            BasicConv2d(in_planes, 128, kernel_size=1, stride=1, padding=0),
            BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        Path1 = self.path1(x)
        Path2 = self.path2(x)
        Path3 = self.path3(x)
        Path4 = self.path4(x)

        out = torch.cat((Path1, Path2, Path3, Path4), 1)
        return out

class featureExtrationA(nn.Module):  # 192, k256/2, l256/2, m192/3, n192/3, p96/3, q192/3

    def __init__(self, in_planes):
        super(featureExtrationA, self).__init__()

        self.path1 = nn.Sequential(
            BasicConv2d(in_planes, 96, kernel_size=1, stride=1, padding=0),
            BasicConv2d(96, 192, kernel_size=7, stride=1, padding=3)
        )

        self.path2 = BasicConv2d(in_planes, 192, kernel_size=3, stride=1, padding=1)

        self.path3 = nn.Sequential(
            BasicConv2d(in_planes, 256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 192, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        out = torch.cat((x1, x2, x3), 1)

        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

# 输入:混合图,遮罩
# 输出:修复后的梯度,边缘
class Gradient_Restruct_Network(nn.Module):

    def __init__(self, model,residual_blocks=8, use_spectral_norm=True):
        super(Gradient_Restruct_Network, self).__init__()
        # ReflectionNetwork
        self.model = model
        # x 128*128
        self.convs1R = nn.Sequential(*self.model[0:7])  # 64*64
        self.convs2R = nn.Sequential(*self.model[7:14])  # 32*32
        self.convs3R = nn.Sequential(*self.model[14:24])  # 16*16
        self.convs4R = nn.Sequential(*self.model[24:34])  # 8*8
        # self.convs5R = nn.Sequential(*self.model[34:44])  # 4*4
        '''
        torch.Size([8, 3, 256, 256])
        torch.Size([8, 64, 128, 128])
        torch.Size([8, 128, 64, 64])
        torch.Size([8, 256, 32, 32])
        torch.Size([8, 512, 16, 16])
        torch.Size([8, 512, 8, 8])
        '''

        self.conv6R = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.featureExtractionA = featureExtrationA(256)

        # GradientNetwork
        # self.convg = nn.Conv2d(3, 48, kernel_size = 5, stride = 1, padding = 2)

        self.conv6_1 = BasicConv2d(512, 1024, kernel_size=7, stride=1, padding=3)
        self.conv6_2 = BasicConv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        #
        self.deconv5_1 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # # self.deconv5_2 = Conv2dUnit(256, 256, kernel_size = 4, stride = 2, padding = 1)
        self.deconv5_2 = BasicConv2d(256, 256, kernel_size=5, stride=1, padding=2)
        # self.featureEnhance5 = BasicConv2d(256, 128, kernel_size=7, stride=1, padding=3)
        # # 32*32
        #
        self.deconv4_1 = BasicConv2d(768, 128, kernel_size=3, stride=1, padding=1)
        self.deconv4_2 = Conv2dUnit(128, 128, kernel_size=4, stride=2, padding=1)
        # self.featureEnhance4 = BasicConv2d(128, 64, kernel_size=7, stride=1, padding=3)
        # # 64*64
        self.deconv3_1 = BasicConv2d(384, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3_2 = Conv2dUnit(64, 64, kernel_size=4, stride=2, padding=1)
        # self.featureEnhance3 = BasicConv2d(64, 32, kernel_size=7, stride=1, padding=3)
        # # 128*128
        self.deconv2_1 = BasicConv2d(192, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2_2 = Conv2dUnit(32, 32, kernel_size=4, stride=2, padding=1)
        # self.featureEnhance2 = BasicConv2d(32, 16, kernel_size=7, stride=1, padding=3)
        # 256*256

        self.deconv1 = Conv2dUnit(96, 64, kernel_size=4, stride=2, padding=1)

        self.pred1_contour = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()


        # 后来自己加的

        self.encoder = nn.Sequential(
            # 按规律填充
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=5, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            # 归一化
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),


        )

        # blocks = []
        # for _ in range(residual_blocks):
        #     block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
        #     blocks.append(block)
        #
        # self.middle = nn.Sequential(*blocks)
        self.middle = nn.Sequential(GlobalContextBlock(inplanes=64, ratio=0.25),

                                    spectral_norm(
                                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                                        use_spectral_norm),
                                    nn.InstanceNorm2d(128, track_running_stats=False),
                                    nn.ReLU(True),

                                    spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2,
                                                            padding=1), use_spectral_norm),
                                    nn.InstanceNorm2d(256, track_running_stats=False),
                                    nn.ReLU(True)
                                    )

        self.decoder_gradient = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

    def forward(self, x,mask):
        # 流程应该是这样的
        images_masked = (x * (1 - mask))
        # AllNetwork
        # convs1R -> convs5R 是VGG特征提取
        convs1R = self.convs1R(images_masked)  # 64*64
        convs2R = self.convs2R(convs1R)  # 32*32
        convs3R = self.convs3R(convs2R)  # 16*16
        convs4R = self.convs4R(convs3R)  # 8*8
        # convs5R = self.convs5R(convs4R)  # 4*4

        # 提取到convs5R 图像特征
        # Gradient Network y 128*128
        convs6g = self.conv6_2(self.conv6_1(convs4R))  # 4*4

        # print(convs6g.size())
        deconv5g = self.deconv5_2(self.deconv5_1(convs6g))  # 8*8
        # print(deconv5g.size())
        # print(convs4R.size())
        sum1g = torch.cat((deconv5g, convs4R), 1)  # 256+512 = 768
        deconv4g = self.deconv4_2(self.deconv4_1(sum1g))  # 16*16
        # sum2g = deconv4g + convs3g
        sum2g = torch.cat((deconv4g, convs3R), 1)  # 128+256 = 384
        deconv3g = self.deconv3_2(self.deconv3_1(sum2g))  # 32*32
        # sum3g = deconv3g + convs2g #64+128 = 192
        sum3g = torch.cat((deconv3g, convs2R), 1)
        deconv2g = self.deconv2_2(self.deconv2_1(sum3g))  # 64*64
        # sum4g = deconv2g+convs1g #32+64 = 96
        sum4g = torch.cat((deconv2g, convs1R), 1)
        deconv1g = self.deconv1(sum4g)  # 128*128
        pred1_gradient = self.pred1_contour(deconv1g)

        # 得到去除反射的破损梯度图
        gradient = self.sigmoid(pred1_gradient)
        # 对梯度图加遮罩
        # gradient = gradient*(1-mask)+mask

        # 加入边缘修复模型，进行结构恢复

        gradient_masked = (gradient * (1 - mask))
        inputs = torch.cat((images_masked, gradient_masked, mask), dim=1)

        # print(inputs.shape)
        # edge = self.encoder(inputs)
        # edge = self.middle(edge)
        # edge = self.decoder_edge(edge)
        # edge = torch.sigmoid(edge)

        # 加入梯度修复模型，进行梯度恢复
        # gradient_edge = gradient_masked+edge
        # inputs = torch.cat((images_masked, gradient_masked, mask), dim=1)
        # gradient = self.encoder(inputs)
        # gradient = self.middle(gradient)
        # gradient = self.decoder_gradient(gradient)
        # gradient = torch.sigmoid(gradient)

        encode = self.encoder(inputs)
        encode = self.middle(encode)

        # edge = self.decoder_edge(encode)
        # edge = torch.sigmoid(edge)

        gradient = self.decoder_gradient(encode)
        gradient = torch.sigmoid(gradient)


        # outputE = edge
        outputG = gradient

        # Reflection network

        # return outputE, outputG
        return  outputG

    def maxout(self, x):
        for step in range(24):
            maxtmp, index = torch.max(x[:, ((step + 1) * 2 - 2):(step + 1) * 2, :, :], dim=1)
            if step == 0:
                F1 = maxtmp.unsqueeze(1)
            else:
                F1 = torch.cat((F1, maxtmp.unsqueeze(1)), 1)
        return F1



# 输入:混合图,遮罩
# 输出:修复后的梯度
# 原始为分段，现在简化为一段
class Gradient_Restruct_Network1(nn.Module):

    def __init__(self, model,residual_blocks=8, use_spectral_norm=True):
        super(Gradient_Restruct_Network1, self).__init__()
        # ReflectionNetwork
        self.model = model
        # x 128*128
        self.convs1R = nn.Sequential(*self.model[0:7])  # 64*64
        self.convs2R = nn.Sequential(*self.model[7:14])  # 32*32
        self.convs3R = nn.Sequential(*self.model[14:24])  # 16*16
        self.convs4R = nn.Sequential(*self.model[24:34])  # 8*8
        # self.convs5R = nn.Sequential(*self.model[34:44])  # 4*4
        '''
        torch.Size([8, 3, 256, 256])
        torch.Size([8, 64, 128, 128])
        torch.Size([8, 128, 64, 64])
        torch.Size([8, 256, 32, 32])
        torch.Size([8, 512, 16, 16])
        torch.Size([8, 512, 8, 8])
        '''
        self.gcn512 = GlobalContextBlock(inplanes=512, ratio=0.25)

        self.conv6R = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.featureExtractionA = featureExtrationA(256)

        # GradientNetwork
        # self.convg = nn.Conv2d(3, 48, kernel_size = 5, stride = 1, padding = 2)

        self.conv6_1 = BasicConv2d(512, 1024, kernel_size=7, stride=1, padding=3)
        self.conv6_2 = BasicConv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        #
        self.deconv5_1 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # # self.deconv5_2 = Conv2dUnit(256, 256, kernel_size = 4, stride = 2, padding = 1)
        self.deconv5_2 = BasicConv2d(256, 256, kernel_size=5, stride=1, padding=2)
        # self.featureEnhance5 = BasicConv2d(256, 128, kernel_size=7, stride=1, padding=3)
        # # 32*32
        #
        self.deconv4_1 = BasicConv2d(768, 128, kernel_size=3, stride=1, padding=1)
        self.deconv4_2 = Conv2dUnit(128, 128, kernel_size=4, stride=2, padding=1)
        # self.featureEnhance4 = BasicConv2d(128, 64, kernel_size=7, stride=1, padding=3)
        # # 64*64
        self.deconv3_1 = BasicConv2d(384, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3_2 = Conv2dUnit(64, 64, kernel_size=4, stride=2, padding=1)
        # self.featureEnhance3 = BasicConv2d(64, 32, kernel_size=7, stride=1, padding=3)
        # # 128*128
        self.deconv2_1 = BasicConv2d(192, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2_2 = Conv2dUnit(32, 32, kernel_size=4, stride=2, padding=1)
        # self.featureEnhance2 = BasicConv2d(32, 16, kernel_size=7, stride=1, padding=3)
        # 256*256

        self.deconv1 = Conv2dUnit(96, 64, kernel_size=4, stride=2, padding=1)

        self.pred1_contour = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()


        # 后来自己加的

        # self.encoder = nn.Sequential(
        #     # 按规律填充
        #     nn.ReflectionPad2d(3),
        #     spectral_norm(nn.Conv2d(in_channels=5, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
        #     # 归一化
        #     nn.InstanceNorm2d(64, track_running_stats=False),
        #     nn.ReLU(True),
        #
        #
        # )

        # blocks = []
        # for _ in range(residual_blocks):
        #     block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
        #     blocks.append(block)
        #
        # self.middle = nn.Sequential(*blocks)
        # self.middle = nn.Sequential(GlobalContextBlock(inplanes=64, ratio=0.25),
        #                             spectral_norm(
        #                                 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
        #                                 use_spectral_norm),
        #                             nn.InstanceNorm2d(128, track_running_stats=False),
        #                             nn.ReLU(True),
        #
        #                             spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2,
        #                                                     padding=1), use_spectral_norm),
        #                             nn.InstanceNorm2d(256, track_running_stats=False),
        #                             nn.ReLU(True)
        #                             )
        #
        # self.decoder_gradient = nn.Sequential(
        #     spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
        #     nn.InstanceNorm2d(128, track_running_stats=False),
        #     nn.ReLU(True),
        #
        #     spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
        #     nn.InstanceNorm2d(64, track_running_stats=False),
        #     nn.ReLU(True),
        #
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        # )

    def forward(self, x,mask):
        # 流程应该是这样的
        images_masked = (x * (1 - mask))
        # AllNetwork
        # convs1R -> convs5R 是VGG特征提取
        convs1R = self.convs1R(images_masked)  # 64*64
        convs2R = self.convs2R(convs1R)  # 32*32
        convs3R = self.convs3R(convs2R)  # 16*16
        convs4R = self.convs4R(convs3R)  # 8*8
        # convs5R = self.convs5R(convs4R)  # 4*4

        # 加入注意力
        convs4R = self.gcn512(convs4R)

        # 提取到convs5R 图像特征
        # Gradient Network y 128*128
        convs6g = self.conv6_2(self.conv6_1(convs4R))  # 4*4

        # print(convs6g.size())
        deconv5g = self.deconv5_2(self.deconv5_1(convs6g))  # 8*8
        # print(deconv5g.size())
        # print(convs4R.size())
        sum1g = torch.cat((deconv5g, convs4R), 1)  # 256+512 = 768
        deconv4g = self.deconv4_2(self.deconv4_1(sum1g))  # 16*16
        # sum2g = deconv4g + convs3g
        sum2g = torch.cat((deconv4g, convs3R), 1)  # 128+256 = 384
        deconv3g = self.deconv3_2(self.deconv3_1(sum2g))  # 32*32
        # sum3g = deconv3g + convs2g #64+128 = 192
        sum3g = torch.cat((deconv3g, convs2R), 1)
        deconv2g = self.deconv2_2(self.deconv2_1(sum3g))  # 64*64
        # sum4g = deconv2g+convs1g #32+64 = 96
        sum4g = torch.cat((deconv2g, convs1R), 1)
        deconv1g = self.deconv1(sum4g)  # 128*128
        pred1_gradient = self.pred1_contour(deconv1g)

        # 得到去除反射的破损梯度图
        gradient = self.sigmoid(pred1_gradient)

        outputG = gradient

        # Reflection network

        # return outputE, outputG
        return  outputG


# 输入:混合图,遮罩
# 输出:修复后的梯度,边缘;去除反射后的图;去除反射且修复遮罩部位的图
# 加入GCNet，简化代码
class Restruct_Guidance_Network1(nn.Module):
    def __init__(self, model, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(Restruct_Guidance_Network1, self).__init__()
        # ReflectionNetwork
        self.model = model
        # x 128*128
        self.convs1R = nn.Sequential(*self.model[0:7])  # 64*64
        self.convs2R = nn.Sequential(*self.model[7:14])  # 32*32
        self.convs3R = nn.Sequential(*self.model[14:24])  # 16*16
        self.convs4R = nn.Sequential(*self.model[24:34])  # 8*8
        self.convs5R = nn.Sequential(*self.model[34:44])  # 4*4
        self.gcn512 = GlobalContextBlock(inplanes=512, ratio=0.25)

        self.conv6R = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.featureExtractionA = featureExtrationA(256)

        self.deconv0R1 = nn.Sequential(
            BasicTransConv2d(576, 256, kernel_size=4, stride=2, padding=1)
        )
        self.deconv0R2 = nn.Sequential(
            BasicTransConv2d(576, 256, kernel_size=4, stride=2, padding=1)
        )
        self.deconv0R3 = nn.Sequential(
            BasicTransConv2d(576, 256, kernel_size=4, stride=2, padding=1)
        )

        self.deconv1R1 = nn.Sequential(
            BasicTransConv2d(1536 - 128, 128, kernel_size=4, stride=2, padding=1)
        )
        self.deconv1R2 = nn.Sequential(
            BasicTransConv2d(1536 - 128, 128, kernel_size=4, stride=2, padding=1)
        )
        self.deconv1R3 = nn.Sequential(
            BasicTransConv2d(1536 - 128, 128, kernel_size=4, stride=2, padding=1)
        )

        self.featureExtractionB = featureExtractionB(768 - 64)

        self.deconv2R1 = nn.Sequential(
            BasicTransConv2d(640, 64, kernel_size=4, stride=2, padding=1)
        )
        self.deconv2R2 = nn.Sequential(
            BasicTransConv2d(640, 64, kernel_size=4, stride=2, padding=1)
        )
        self.deconv2R3 = nn.Sequential(
            BasicTransConv2d(640, 64, kernel_size=4, stride=2, padding=1)
        )

        self.deconv3R1 = nn.Sequential(
            BasicTransConv2d(384 - 32, 32, kernel_size=4, stride=2, padding=1)
        )
        self.deconv3R2 = nn.Sequential(
            BasicTransConv2d(384 - 32, 32, kernel_size=4, stride=2, padding=1)
        )
        self.deconv3R3 = nn.Sequential(
            BasicTransConv2d(384 - 32, 32, kernel_size=4, stride=2, padding=1)
        )

        self.deconv4R1 = nn.Sequential(
            BasicTransConv2d(192 - 16, 16, kernel_size=4, stride=2, padding=1)
        )
        self.deconv4R2 = nn.Sequential(
            BasicTransConv2d(192 - 16, 16, kernel_size=4, stride=2, padding=1)
        )
        self.deconv4R3 = nn.Sequential(
            BasicTransConv2d(192 - 16, 16, kernel_size=4, stride=2, padding=1)
        )

        self.output = nn.Sequential(
            BasicConv2d(49, 16, kernel_size=3, stride=1, padding=1),
            BasicConv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )

        self.convg = nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2)

        self.conv6_1 = BasicConv2d(512, 1024, kernel_size=7, stride=1, padding=3)
        self.conv6_2 = BasicConv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        #
        self.deconv5_1 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # # self.deconv5_2 = Conv2dUnit(256, 256, kernel_size = 4, stride = 2, padding = 1)
        self.deconv5_2 = BasicConv2d(256, 256, kernel_size=5, stride=1, padding=2)
        self.featureEnhance5 = BasicConv2d(256, 128, kernel_size=7, stride=1, padding=3)
        # # 32*32
        #
        self.deconv4_1 = BasicConv2d(768, 128, kernel_size=3, stride=1, padding=1)
        self.deconv4_2 = Conv2dUnit(128, 128, kernel_size=4, stride=2, padding=1)
        self.featureEnhance4 = BasicConv2d(128, 64, kernel_size=7, stride=1, padding=3)
        # # 64*64
        self.deconv3_1 = BasicConv2d(384, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3_2 = Conv2dUnit(64, 64, kernel_size=4, stride=2, padding=1)
        self.featureEnhance3 = BasicConv2d(64, 32, kernel_size=7, stride=1, padding=3)
        # # 128*128
        self.deconv2_1 = BasicConv2d(192, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2_2 = Conv2dUnit(32, 32, kernel_size=4, stride=2, padding=1)
        self.featureEnhance2 = BasicConv2d(32, 16, kernel_size=7, stride=1, padding=3)
        # 256*256

        self.deconv1 = Conv2dUnit(96, 64, kernel_size=4, stride=2, padding=1)

        self.pred1_gradient = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)

        self.sigmoid = nn.Sigmoid()

        # # 后来自己加的
        # self.encoder = nn.Sequential(
        #     # 边界反射填充张量
        #     nn.ReflectionPad2d(3),
        #     # 卷积
        #     nn.Conv2d(in_channels=5, out_channels=64, kernel_size=7, padding=0),
        #     # 批量归一化
        #     nn.InstanceNorm2d(64, track_running_stats=False),
        #     nn.ReLU(True),
        #
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
        #     nn.InstanceNorm2d(128, track_running_stats=False),
        #     nn.ReLU(True),
        #
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
        #     nn.InstanceNorm2d(256, track_running_stats=False),
        #     nn.ReLU(True)
        # )
        #
        # blocks = []
        # for _ in range(residual_blocks):
        #     block = ResnetBlock(256, 2)
        #     blocks.append(block)
        #
        # # 残差块
        # # 返回值 = 输入值 + 计算结果
        # self.middle = nn.Sequential(*blocks)
        #
        # self.decoder = nn.Sequential(
        #     # nn.ConvTranspose2d(in_channels=384, out_channels=256, kernel_size=4, stride=2, padding=1),
        #     # nn.InstanceNorm2d(128, track_running_stats=False),
        #     # nn.ReLU(True),
        #
        #     nn.ConvTranspose2d(in_channels=384, out_channels=128, kernel_size=4, stride=2, padding=1),
        #     nn.InstanceNorm2d(128, track_running_stats=False),
        #     nn.ReLU(True),
        #
        #     nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
        #     nn.InstanceNorm2d(64, track_running_stats=False),
        #     nn.ReLU(True),
        #
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        # )


    def forward(self, x, mask):
        images_masked = (x * (1 - mask))
        # AllNetwork
        # convs1R -> convs5R 是VGG特征提取

        convs1R = self.convs1R(images_masked)  # 64*64
        convs2R = self.convs2R(convs1R)  # 32*32
        convs3R = self.convs3R(convs2R)  # 16*16
        convs4R = self.convs4R(convs3R)  # 8*8
        # 加入注意力
        convs4R = self.gcn512(convs4R)
        convs5R = self.convs5R(convs4R)  # 4*4
        # print("{}\n{}\n{}\n{}\n{}\n{}\n".format(images_masked.shape,convs1R.shape,convs2R.shape,convs3R.shape,convs4R.shape,convs5R.shape))
        '''
        torch.Size([8, 3, 256, 256])
        torch.Size([8, 64, 128, 128])
        torch.Size([8, 128, 64, 64])
        torch.Size([8, 256, 32, 32])
        torch.Size([8, 512, 16, 16])
        torch.Size([8, 512, 8, 8])
        '''
        # 提取到convs5R 图像特征
        # Gradient Network y 128*128
        convs6g = self.conv6_2(self.conv6_1(convs4R))  # 4*4

        # print(convs6g.size())
        deconv5g = self.deconv5_2(self.deconv5_1(convs6g))  # 8*8
        # print(deconv5g.size())
        # print(convs4R.size())
        sum1g = torch.cat((deconv5g, convs4R), 1)  # 256+512 = 768
        deconv4g = self.deconv4_2(self.deconv4_1(sum1g))  # 16*16
        # sum2g = deconv4g + convs3g
        sum2g = torch.cat((deconv4g, convs3R), 1)  # 128+256 = 384
        deconv3g = self.deconv3_2(self.deconv3_1(sum2g))  # 32*32
        # sum3g = deconv3g + convs2g #64+128 = 192
        sum3g = torch.cat((deconv3g, convs2R), 1)
        deconv2g = self.deconv2_2(self.deconv2_1(sum3g))  # 64*64
        # sum4g = deconv2g+convs1g #32+64 = 96
        sum4g = torch.cat((deconv2g, convs1R), 1)
        deconv1g = self.deconv1(sum4g)  # 128*128
        pgradient = self.pred1_gradient(deconv1g)

        gradient = self.sigmoid(pgradient)


        # contour = gradient
        # contour = pred1_contour
        #
        # Reflection
        # 将图像特征卷积，生成新的特征
        conv6R = self.conv6R(convs5R)
        featureA = self.featureExtractionA(conv6R)

        #
        deconv01R = self.deconv0R1(featureA)
        deconv02R = self.deconv0R2(featureA)
        deconv03R = self.deconv0R3(featureA)
        # 8*8 256, 256,256, 512
        deconv5gfE = self.featureEnhance5(deconv5g)  # -12
        deconv0R = torch.cat((deconv01R, deconv02R, deconv03R, convs4R, deconv5gfE), 1)

        # deconv0R = deconv0R

        deconv11R = self.deconv1R1(deconv0R)
        deconv12R = self.deconv1R2(deconv0R)
        deconv13R = self.deconv1R3(deconv0R)
        # 16*16 128,128,128, 256
        deconv4gfE = self.featureEnhance4(deconv4g)  # -64
        deconv1R = torch.cat((deconv11R, deconv12R, deconv13R, convs3R, deconv4gfE), 1)
        featureB = self.featureExtractionB(deconv1R)

        deconv21R = self.deconv2R1(featureB)
        deconv22R = self.deconv2R2(featureB)
        deconv23R = self.deconv2R3(featureB)
        # 32*32 64, 64, 64, 128
        deconv3gfE = self.featureEnhance3(deconv3g)  # -32
        deconv2R = torch.cat((deconv21R, deconv22R, deconv23R, convs2R, deconv3gfE), 1)
        deconv2R = deconv2R
        deconv31R = self.deconv3R1(deconv2R)
        deconv32R = self.deconv3R2(deconv2R)
        deconv33R = self.deconv3R3(deconv2R)
        # 64*64	32 32 32 64
        deconv2gfE = self.featureEnhance2(deconv2g)  # -16
        deconv3R = torch.cat((deconv31R, deconv32R, deconv33R, convs1R, deconv2gfE), 1)
        deconv3R = deconv3R
        deconv41R = self.deconv4R1(deconv3R)
        deconv42R = self.deconv4R2(deconv3R)
        deconv43R = self.deconv4R3(deconv3R)
        # 128*128
        deconv4R = torch.cat((deconv41R, deconv42R, deconv43R, gradient), 1)
        deconv4R = deconv4R

        # outputR:remove反射后的输出
        outputR = self.output(deconv4R)
        # outputR_masked = outputR * (1 - mask)


        # # 这里要优化,用原图的话,没有用到去反射
        # # inputs = torch.cat((images_masked,outputR_masked,edge,mask),dim = 1)
        # inputs = torch.cat((outputR_masked, gradient, mask), dim=1)
        # # print("\ninputs:",inputs.shape)
        # outputI = self.encoder(inputs)
        # # print("\nencoder:",outputI.shape)
        # outputI = self.middle(outputI)
        # # print("\nmiddle:",outputI.shape)
        # # print("\noutputI:", outputI.shape," convs2R:",convs2R.shape)
        # outputI = torch.cat((outputI, convs2R), dim=1) # 8*384*64*64
        # outputI = self.decoder(outputI)
        # # print("\ndecoder:",outputI.shape)
        #
        # outputI = torch.sigmoid(outputI)
        '''
            inputs: torch.Size([8, 5, 256, 256])
            encoder: torch.Size([8, 256, 64, 64])
            middle: torch.Size([8, 256, 64, 64])
            decoder: torch.Size([8, 3, 256, 256])
        '''

        return outputR,images_masked,gradient

    def maxout(self, x):
        for step in range(24):
            maxtmp, index = torch.max(x[:, ((step + 1) * 2 - 2):(step + 1) * 2, :, :], dim=1)
            if step == 0:
                F1 = maxtmp.unsqueeze(1)
            else:
                F1 = torch.cat((F1, maxtmp.unsqueeze(1)), 1)
        return F1