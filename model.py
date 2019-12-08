import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import utils

class FFDNet(nn.Module):

    def __init__(self, is_gray):
        super(FFDNet, self).__init__()

        if is_gray: # 灰度
            self.num_conv_layers = 15 # 所有卷积层数
            self.downsampled_channels = 5 # 下采样后输入通道
            self.num_feature_maps = 64 # 中间输出特征
            self.output_features = 4 # 最终输出通道
        else: # 彩色
            self.num_conv_layers = 12
            self.downsampled_channels = 15
            self.num_feature_maps = 96
            self.output_features = 12
            
        self.kernel_size = 3 # 卷积核大小
        self.padding = 1 # 卷积填充大小
        
        layers = []
        # Conv + Relu
        layers.append(nn.Conv2d(in_channels=self.downsampled_channels, out_channels=self.num_feature_maps, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Conv + BN + Relu
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.num_feature_maps, \
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))
            layers.append(nn.BatchNorm2d(self.num_feature_maps))
            layers.append(nn.ReLU(inplace=True))
        
        # Conv
        layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.output_features, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        self.itermediate_dncnn = nn.Sequential(*layers)

    def forward(self, x, noise_sigma):
        x_cat = utils.downsample(x.data, noise_sigma.data)
        x_cat = Variable(x_cat)
        h_dncnn = self.intermediate_dncnn(x_cat)
        y_pred = utils.upsample(h_dncnn)
        return y_pred
