import numpy as np
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from skimage.measure.simple_metrics import compare_psnr

def is_image_gray(image):
    """
    判断 cvImage 是否为灰度
    """
    # a[..., 0] == a.T[0].T
    return not(len(im.shape) == 3 and not(np.allclose(im[...,0], im[...,1]) and np.allclose(im[...,2], im[...,1])))

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
    sca2 = sca * sca
    Cout = sca2 * C
    Hout = H // sca
    Wout = W // sca
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

def upsample(input):
    """
    将卷积完的结果上采样
    """
    N, Cin, Hin, Win = input.size()
    dtype = input.type()
    sca = 2
    sca2 = sca * sca
    Cout = Cin // sca2
    Hout = Hin * sca
    Wout = Win * sca
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    assert (Cin % sca2 == 0), 'Invalid input dimensions: number of channels should be divisible by 4'

    result = torch.zeros((N, Cout, Hout, Wout)).type(dtype)
    for idx in range(sca2):
        result[:, :, idxL[idx][0] :: sca, idxL[idx][1] :: sca] = input[:, idx : Cin : sca2, :, :]

    return result

def normalize(data):
    """
    图像归一化
    """
    return np.float32(data / 255)

def image_to_patches(image, patch_size):
    """
    将图片转化成区域集
    :param image: Image (C * W * H) Numpy
    :param patch_size: int
    :return: (patch_num, C, win, win)
    """
    W = image.shape[1]
    H = image.shape[2]
    if W < patch_size or H < patch_size:
        return []

    ret = []
    for ws in range(0, W // patch_size):
        for hs in range(0, H // patch_size):
            patch = image[:, ws * patch_size : (ws + 1) * patch_size, hs * patch_size : (hs + 1) * patch_size]
            ret.append(patch)
    return np.array(ret, dtype=np.float32)

def add_batch_noise(images, noise_sigma):
    """
    往一个批次图像中 添加相同的噪声等级
    :param images: Image (n, C, W, H) Tensor
    :return: Image (n, C, W, H)
    """
    new_images = []
    for image in images:
        noise = np.random.random(image.shape) * noise_sigma
        new_images.append(image.numpy() + noise)
    return torch.FloatTensor(new_images)

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
        res = (varim.data.cpu().numpy()[0, 0, :] * 255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res*255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res

def weights_init_kaiming(lyr):
    """
    Initializes weights of the model according to the "He" initialization
    method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
    This function is to be called by the torch.nn.Module.apply() method,
    which applies weights_init_kaiming() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
            clamp_(-0.025, 0.025)
        nn.init.constant_(lyr.bias.data, 0.0)