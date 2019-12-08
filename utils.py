import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from skimage.measure.simple_metrics import compare_psnr

def downsample(input, noise_sigma):
    """
    将输入图像下采样为四张小图
    :param input: C * H * W
    :param noise_sigma: C * H/2 * W/2
    :return: 4 * C * H/2 * W/2
    """
    # noise_sigma is a list of length batch_size
    N, C, H, W = input.size()
    dtype = input.type()
    sca = 2
    sca2 = sca*sca
    Cout = sca2*C
    Hout = H//sca
    Wout = W//sca
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # Fill the downsampled image with zeros
    if 'cuda' in dtype:
        downsampledfeatures = torch.cuda.FloatTensor(N, Cout, Hout, Wout).fill_(0)
    else:
        downsampledfeatures = torch.FloatTensor(N, Cout, Hout, Wout).fill_(0)

    # Build the CxH/2xW/2 noise map
    noise_map = noise_sigma.view(N, 1, 1, 1).repeat(1, C, Hout, Wout)

    # Populate output
    for idx in range(sca2):
        downsampledfeatures[:, idx:Cout:sca2, :, :] = input[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]

    # concatenate de-interleaved mosaic with noise map
    return torch.cat((noise_map, downsampledfeatures), 1)

def upsample(x):
    """
    将卷积完的结果上采样
    """
    N, Cin, Hin, Win = input.size()
    dtype = input.type()
    sca = 2
    sca2 = sca*sca
    Cout = Cin//sca2
    Hout = Hin*sca
    Wout = Win*sca
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    assert (Cin%sca2 == 0), 'Invalid input dimensions: number of channels should be divisible by 4'

    result = torch.zeros((N, Cout, Hout, Wout)).type(dtype)
    for idx in range(sca2):
        result[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca] = input[:, idx:Cin:sca2, :, :]

    return result

def normalize(data):
    """
    图像归一化
    """
    return np.float32(data / 255)

def add_batch_noise(images, noise_sigma):
    """
    往一个批次图像中 添加相同的噪声等级
    :param images: Image (n, C, W, H)
    :return: Image (n, C, W, H)
    """
    new_images = []
    for image in images:
        noise = torch.FloatTensor(image.size()).normal_(mean=0, std=noise_sigma)
        new_images.append(image + noise)
    new_images = Variable(np.array(new_images, dtype=np.float32))
    return new_images

def batch_psnr(img, imclean, data_range):
    """
    计算整个批次的 PSNR
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range)
    return psnr/img_cpu.shape[0]

def variable_to_cv2_image(varim):
    """
    Variable -> Cv2
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res*255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res

def weights_init_kaiming(lyr):
	"""Initializes weights of the model according to the "He" initialization
	method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
	This function is to be called by the torch.nn.Module.apply() method,
	which applies weights_init_kaiming() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
			clamp_(-0.025, 0.025)
		nn.init.constant(lyr.bias.data, 0.0)