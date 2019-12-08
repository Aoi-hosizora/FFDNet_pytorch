import argparse
import numpy as np
import cv2
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import FFDNet
import utils

def read_image(image_path) -> ([], bool):
    """
    根据路径获取图像，并处理形状与归一化
    :return: Normalized Image (C * W * H), is_gray
    """
    # a[..., 0] == a.T[0].T
    im = cv2.imread(args.test_path)
    is_gray = len(im.shape) == 3 and not(np.allclose(im[...,0], im[...,1]) and np.allclose(im[...,2], im[...,1]))
    print("{} image shape: {}".format("Gray" if is_gray else "RGB", im.shape))

    if is_gray:
        model_path = args.model_path + 'net_gray.pth'
        image = cv2.imread(args.test_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image.T, 0) # 1 * W * H
    else:
        model_path = args.model_path + 'net_rgb.pth'
        image = cv2.imread(args.test_path)
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).transpose(2, 1, 0) # 3 * W * H
    
    return utils.normalize(image), is_gray

def load_images(is_train, is_gray, base_path):
    """
    加载训练集或验证集图像
    :param base_path: 根目录 ./train_data/
    :return: Images (n * C * W * H)
    """
    if is_gray:
        train_dir = 'gray/train/'
        val_dir = 'gray/val/'
    else:
        train_dir = 'rgb/train/'
        val_dir = 'rgb/val/'
    
    image_dir = base_path + (train_dir if is_train else val_dir)
    images = []
    for _, _, fn in os.walk(image_dir):
        image, image_gray = read_image(args.test_path)
        assert image_gray == is_gray, "图像颜色与训练参数不符合"
        images.append(image)
    return np.array(images, dtype=np.float32)

def train(args):
    # Data
    print('Loading dataset ...')
    train_dataset = load_images(is_train=True, is_gray=args.is_gray, base_path=args.train_path)
    val_dataset = load_images(is_train=False, is_gray=args.is_gray, base_path=args.train_path)
    
    train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
    val_dataset = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)
    print(f'Train dataset: {len(train_dataset)}')
    print(f'Val dataset: {len(val_dataset)}')

    # Model & Optim
    model = FFDNet(is_gray=is_gray)
    model.apply(utils.weights_init_kaiming)
    if args.use_gpu:
        model = model.cuda()
	loss_fn = nn.MSELoss()
	optimizer = optim.Adam(lr=args.learning_rate)

    train_noises = args.train_noise_interval # [0, 75, 15]
    val_noises = args.val_noise_interval # [0, 60, 20]

    for epoch_idx in range(args.epoches):
        # Train
        model.train()
        for batch_data in train_dataset:
            # According to internal, add noise
            for noise_sigma in range(train_noises[0], train_noises[1], train_noises[2]):
                new_images = utils.add_batch_noise(batch_data, noise_sigma)
                if args.use_gpu:
                    new_images = new_images.cuda()

                # Predict
                images_pred = model(new_images, noise_sigma)
                train_loss = loss_fn(images_pred, batch_data)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        loss_idx = 0
        for batch_data in val_dataset:
            # According to internal, add noise
            for noise_sigma in range(val_noises[0], val_noises[1], val_noises[2]):
                new_images = utils.add_batch_noise(batch_data, noise_sigma)
                if args.use_gpu:
                    new_images = new_images.cuda()
                
                # Predict
                images_pred = model(new_images, noise_sigma)
                val_loss += loss_fn(images_pred, batch_data)
                loss_idx += 1
                
        val_loss /= loss_idx
        print(f'Epoch: {epoch_idx}, Train_Loss: {train_loss}, Val_Loss: {val_loss}')

        # Save Checkpoint
        if (epoch_idx + 1) % save_checkpoints == 0:
            model_path = args.model_path + 'net_gray_checkpoint.pth' if is_gray else 'net_rgb_checkpoint.pth'
            torch.save(model.state_dict(), model_path)
            print(f'Saved Checkpoint at Epoch {epoch_idx} in {model_path}')

    # Final Save Model Dict
    model_path = args.model_path + 'net_gray.pth' if is_gray else 'net_rgb.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Saved State Dict in {model_path}')

def test(args):
    # Image
    image, is_gray = read_image(args.test_path)
    print("{} image shape: {}".format("Gray" if is_gray else "RGB", im.shape))
    image = np.expand_dims(image, 0) # 1 * C(1 / 3) * W * H
    image = torch.FloatTensor(image)

    # Noise
    noise_sigma = torch.FloatTensor([args.noise_sigma])
    if args.add_noise:
        image = utils.add_batch_noise(image, arg.noise_sigma)

    # Model & GPU
    model = FFDNet(is_gray=is_gray)
    if args.use_gpu:
        image = image.cuda()
        noise_sigma = noise_sigma.cuda()
        model = model.cuda()

    # Dict
    print("Loading model param...")
    model_path = args.model_path + 'net_gray.pth' if is_gray else 'net_rgb.pth'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Test
    with torch.no_grad():
        start_time = time.time()
        image_pred = model(image, noise_sigma)
        stop_time = time.time()
        print("Test time: {0:.4f}s".format(stop_time - start_time))

    # PSNR
    psnr = utils.batch_psnr(image_pred, image, 1)
    print("PSNR denoised {0:.2f}dB".format(psnr))

    # Save
    cv2.imwrite("ffdnet.png", utils.variable_to_cv2_image(image_pred))
    if args.add_noise:
        cv2.imwrite("noisy.png", utils.variable_to_cv2_image(image))

def main():
    parser = argparse.ArgumentParser()

    # Train
    parser.add_argument("--train_path", type=str, default='./train_data/',                  help='图像存储路径')
    parser.add_argument("--is_gray", action='store_true',                                   help='训练灰度图像')
    parser.add_argument("--train_noise_interval", nargs=3, type=int, default=[0, 75, 15],   help='训练的噪声间隔')
    parser.add_argument("--val_noise_interval", nargs=3, type=int, default=[0, 60, 20],     help='验证的噪声间隔')
    parser.add_argument("--batch_size", type=int, default=128,                              help='批次大小')
    parser.add_argument("--epoches", type=int, default=80,                                  help='训练轮次数')
    parser.add_argument("--learning_rate", type=float, default=1e-3,                        help='Adam 初始学习率')
    parser.add_argument("--save_checkpoints", type=int, default=5,                          help='多少个轮次保存检查点')

    # Test
    parser.add_argument("--test_path", type=str, default='./test_data/',                    help='测试图片路径')
    parser.add_argument("--noise_sigma", type=float, default=25,                            help='测试输入的噪声等级')
    parser.add_argument('--add_noise', action='store_true',                                 help='是否添加噪声')

    # Global
    parser.add_argument("--model_path", type=str, default='./models/',                      help='保存或者测试用的模型路径')
    parser.add_argument("--use_gpu", action='store_true',                                   help='是否使用 GPU')
    parser.add_argument("--is_train", action='store_true',                                  help='是否训练')
    parser.add_argument("--is_test", action='store_true',                                   help='是否测试')

    args = parser.parse_args()
    assert (args.is_train or args.is_test), 'is_train 和 is_test 至少有一个为 True'

    # Normalize noise level
    args.train_noise_interval[0] /= 255
    args.train_noise_interval[1] /= 255
    args.train_noise_interval[2] /= 255
    args.val_noise_interval[0] /= 255
    args.val_noise_interval[1] /= 255
    args.val_noise_interval[2] /= 255
    args.val_noise /= 255
    args.noise_sigma /= 255

    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    print("> Parameters: ")
    for k, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print(f'\t{k}: {v}')
    print('\n')

    if args.is_train:
        train(args)

    if args.is_test:
        test(args)

if __name__ == "__main__":
    main()
