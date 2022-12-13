import argparse
import math

import numpy as np
import torch
import cv2
import PIL.Image as Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MyDataSet
from models.IRCNN import IRCNN
from train import resume
from train import evaluate
from models.unet_model import UNet
from train_unet import evaluate as ev2
import glob

from utils.process_raw_images import process_raw_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set noise level
noise_level = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='ircnn_color_image_denoiser_15')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--show_image', type=int, default=1, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.ckpt = 'unet_color_image_denoiser_{n}'.format(n = noise_level)
    img_paths = ['C:/Users/Administrator/Desktop/adcv/Learning deep CNN denoiser prior for image restoration/data/test/CBSD68/*'
                 ]
    out_put_path = '//data/test/CBSD68_patches'

    process_raw_images(img_paths, out_put_path, noise_level)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataloaders
    dataset = MyDataSet('./data/train')
    test_set = MyDataSet('./data/test/CBSD68_patches')
    trainloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2)
    testloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2)
    dataloaders = (trainloader, testloader)

    # network
    model = UNet(3,1).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # resume the trained model
    model, optimizer = resume(args, model, optimizer)
    print('Average PSNR with noise level {n} : {p}'.format(n=noise_level, p=ev2(model, testloader)))


