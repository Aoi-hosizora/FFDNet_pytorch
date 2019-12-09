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

def read_image(image_path, is_gray):
    """
    根据路径获取图像，并处理形状与归一化
    :return: Normalized Image (C * W * H)
    """
    if is_gray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image.T, 0) # 1 * W * H
    else:
        image = cv2.imread(image_path)
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).transpose(2, 1, 0) # 3 * W * H
    
    return utils.normalize(image)

def load_images(is_train, is_gray, base_path):
    """
    加载训练集或验证集图像
    :param base_path: 根目录 ./train_data/
    :return: List[Patches] (C * W * H)
    """
    if is_gray:
        train_dir = 'gray/train/'
        val_dir = 'gray/val/'
    else:
        train_dir = 'rgb/train/'
        val_dir = 'rgb/val/'
    
    image_dir = base_path + (train_dir if is_train else val_dir)
    images = []
    for fn in next(os.walk(image_dir))[2]:
        image = read_image(image_dir + fn, is_gray)
        images.append(image)
    return images

def images_to_patches(images, patch_size):
    """
    将图片集合转化成等大的区域集合
    :param images: List[Image (C * W * H)]
    :param patch_size: int
    :return: (n * C * W * H)
    """
    patches_list = []
    for image in images:
        patches = utils.image_to_patches(image, patch_size=patch_size)
        if len(patches) != 0:
            patches_list.append(patches)
    del images
    return np.vstack(patches_list)

def train(args):
    print('> Loading dataset...')
    # Images
    train_dataset = load_images(is_train=True, is_gray=args.is_gray, base_path=args.train_path)
    val_dataset = load_images(is_train=False, is_gray=args.is_gray, base_path=args.train_path)
    print(f'\tTrain image datasets: {len(train_dataset)}')
    print(f'\tVal image datasets: {len(val_dataset)}')

    # Patches
    train_dataset = images_to_patches(train_dataset, patch_size=args.patch_size)
    val_dataset = images_to_patches(val_dataset, patch_size=args.patch_size)
    print(f'\tTrain patch datasets: {train_dataset.shape}')
    print(f'\tVal patch datasets: {val_dataset.shape}')

    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)
    print(f'\tTrain batch number: {len(train_dataloader)}')
    print(f'\tVal batch number: {len(val_dataloader)}')

    # Noise list
    train_noises = args.train_noise_interval # [0, 75, 15]
    val_noises = args.val_noise_interval # [0, 60, 30]
    train_noises = list(range(train_noises[0], train_noises[1], train_noises[2]))
    val_noises = list(range(val_noises[0], val_noises[1], val_noises[2]))
    print(f'\tTrain noise internal: {train_noises}')
    print(f'\tVal noise internal: {val_noises}')
    print('\n')

    # Model & Optim
    model = FFDNet(is_gray=args.is_gray)
    model.apply(utils.weights_init_kaiming)
    if args.cuda:
        model = model.cuda()
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print('> Start training...')
    for epoch_idx in range(args.epoches):
        # Train
        model.train()
        for batch_idx, batch_data in enumerate(train_dataloader):
            # According to internal, add noise
            for int_noise_sigma in train_noises:
                start_time = time.time()

                noise_sigma = int_noise_sigma / 255
                new_images = utils.add_batch_noise(batch_data.cpu(), noise_sigma)
                noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(new_images.shape[0])]))
                new_images = Variable(new_images)
                noise_sigma = Variable(noise_sigma)
                if args.cuda:
                    new_images = new_images.cuda()
                    noise_sigma = noise_sigma.cuda()
                    batch_data = batch_data.cuda()

                # Predict
                images_pred = model(new_images, noise_sigma)
                train_loss = loss_fn(images_pred, batch_data)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # Log Progress
                stop_time = time.time()
                all_num = len(train_dataloader) * len(train_noises)
                done_num = batch_idx * len(train_noises) + train_noises.index(int_noise_sigma) + 1
                rest_time = int((stop_time - start_time) * (all_num - done_num))
                percent = int(done_num / all_num * 100)
                print(f'\rEpoch: {epoch_idx + 1} / {args.epoches}, ' +
                      f'Batch: {batch_idx + 1} / {len(train_dataloader)}, ' +
                      f'Noise_Sigma: {int_noise_sigma} / {train_noises[-1]}, ' +
                      f'Train_Loss: {train_loss}, ' +
                      f'=> {rest_time}s, {percent}%', end='')
        
        print()
        # Evaluate
        model.eval()
        val_loss = 0
        loss_idx = 0
        for batch_idx, batch_data in enumerate(val_dataloader):
            # According to internal, add noise
            for int_noise_sigma in val_noises:
                start_time = time.time()

                noise_sigma = int_noise_sigma / 255
                new_images = utils.add_batch_noise(batch_data.cpu(), noise_sigma)
                noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(new_images.shape[0])]))
                new_images = Variable(new_images)
                noise_sigma = Variable(noise_sigma)
                if args.cuda:
                    new_images = new_images.cuda()
                    noise_sigma = noise_sigma.cuda()
                    batch_data = batch_data.cuda()
                
                # Predict
                images_pred = model(new_images, noise_sigma)
                val_loss += loss_fn(images_pred, batch_data)
                loss_idx += 1
                
                # Log Progress
                stop_time = time.time()
                all_num = len(val_dataloader) * len(val_noises)
                done_num = batch_idx * len(val_noises) + val_noises.index(int_noise_sigma) + 1
                rest_time = int((stop_time - start_time) * (all_num - done_num))
                percent = int(done_num / all_num * 100)
                print(f'\rEpoch: {epoch_idx + 1} / {args.epoches}, ' +
                      f'Batch: {batch_idx + 1} / {len(val_dataloader)}, ' +
                      f'Noise_Sigma: {int_noise_sigma} / {val_noises[-1]}, ' +
                      f'Val_Loss: {val_loss}, ' + 
                      f'=> {rest_time}s, {percent}%', end='')
                
        val_loss /= loss_idx
        print(f'\n| Epoch: {epoch_idx}, Train_Loss: {train_loss}, Val_Loss: {val_loss}')

        # Save Checkpoint
        if (epoch_idx + 1) % args.save_checkpoints == 0:
            model_path = args.model_path + ('net_gray_checkpoint.pth' if args.is_gray else 'net_rgb_checkpoint.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Saved Checkpoint at Epoch {epoch_idx} in {model_path}')

    # Final Save Model Dict
    model_path = args.model_path + ('net_gray.pth' if args.is_gray else 'net_rgb.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Saved State Dict in {model_path}')

def test(args):
    # Image
    image = cv2.imread(args.test_path)
    if image is None:
        raise Exception(f'File {image_path} not found or error')
    is_gray = utils.is_image_gray(image)
    image = read_image(args.test_path, is_gray)
    print("{} image shape: {}".format("Gray" if is_gray else "RGB", image.shape))
    image = np.expand_dims(image, 0) # 1 * C(1 / 3) * W * H
    image = torch.FloatTensor(image)

    # Noise
    if args.add_noise:
        image = utils.add_batch_noise(image, args.noise_sigma)
    noise_sigma = torch.FloatTensor([args.noise_sigma])

    # Model & GPU
    model = FFDNet(is_gray=is_gray)
    if args.cuda:
        image = image.cuda()
        noise_sigma = noise_sigma.cuda()
        model = model.cuda()

    # Dict
    print("> Loading model param...")
    model_path = args.model_path + ('net_gray.pth' if is_gray else 'net_rgb.pth')
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
    parser.add_argument("--patch_size", type=int, default=32,                               help='训练集分割的区域大小')
    parser.add_argument("--train_noise_interval", nargs=3, type=int, default=[0, 75, 15],   help='训练的噪声间隔')
    parser.add_argument("--val_noise_interval", nargs=3, type=int, default=[0, 60, 30],     help='验证的噪声间隔')
    parser.add_argument("--batch_size", type=int, default=256,                              help='批次大小')
    parser.add_argument("--epoches", type=int, default=80,                                  help='训练轮次数')
    parser.add_argument("--learning_rate", type=float, default=1e-3,                        help='Adam 初始学习率')
    parser.add_argument("--save_checkpoints", type=int, default=5,                          help='多少个轮次保存检查点')

    # Test
    parser.add_argument("--test_path", type=str, default='./test_data/color.png',           help='测试图片路径')
    parser.add_argument("--noise_sigma", type=float, default=25,                            help='测试输入的噪声等级')
    parser.add_argument('--add_noise', action='store_true',                                 help='是否添加噪声')

    # Global
    parser.add_argument("--model_path", type=str, default='./models/',                      help='保存或者测试用的模型路径')
    parser.add_argument("--use_gpu", action='store_true',                                   help='是否使用 GPU')
    parser.add_argument("--is_train", action='store_true',                                  help='是否训练')
    parser.add_argument("--is_test", action='store_true',                                   help='是否测试')

    args = parser.parse_args()
    assert (args.is_train or args.is_test), 'is_train 和 is_test 至少有一个为 True'

    args.cuda = args.use_gpu and torch.cuda.is_available()
    print("> Parameters: ")
    for k, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print(f'\t{k}: {v}')
    print('\n')

    # Normalize noise level
    args.noise_sigma /= 255
    args.train_noise_interval[1] += 1
    args.val_noise_interval[1] += 1

    if args.is_train:
        train(args)

    if args.is_test:
        test(args)

if __name__ == "__main__":
    main()
