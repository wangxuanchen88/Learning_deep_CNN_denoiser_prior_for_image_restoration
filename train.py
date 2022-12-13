


import argparse
import time
import os
import cv2

from torch.utils.data import DataLoader, random_split

from utils.evaluation import calculate_psnr
from dataset import MyDataSet
from torchvision import transforms

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from models.IRCNN import IRCNN
from utils.process_raw_images import process_raw_images

"""
This file is used to train the CNN denoisers with new loss function
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VGG_19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
VGG_19 = nn.Sequential(VGG_19.features, VGG_19.avgpool).to(device)
for parameter in VGG_19.parameters():
    parameter.requires_grad = False
vgg_preprocess = models.VGG19_Weights.IMAGENET1K_V1.transforms

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

u = 0.01


def train(args, model, optimizer, dataloaders):
    trainloader, validationloader = dataloaders

    maximum_validation_psrn = 0

    # training
    print('Network training starts ...')
    for epoch in range(args.epochs):
        model.train()

        batch_time = time.time()
        iter_time = time.time()

        for i, data in enumerate(trainloader):
            noised_image = data['noised'].to(device)
            target = data['target'].to(device)
            noise, denoised_image = model(noised_image)

            a = torch.mean((noise - (noised_image - target)) ** 2, dim=[1, 2, 3])
            loss1 = torch.sum(a) / (2 * denoised_image.shape[0])
            loss2 = F.mse_loss(VGG_19(trans(target)), VGG_19(trans(denoised_image)))
            loss = loss1 + u * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            if i % 100 == 0 and i != 0:
                print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                                                                           time.time() - iter_time, loss.item()))
                iter_time = time.time()
        batch_time = time.time() - batch_time
        print('[epoch {} | time:{:.2f} | loss:{:.5f}]'.format(epoch, batch_time, loss.item()))

        # evaluation
        if epoch % 1 == 0:
            validation_psrn = evaluate(model, validationloader)
            print('validation psnr: {:.3f}'.format(validation_psrn))
            print('-------------------------------------------------')

            if validation_psrn > maximum_validation_psrn:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, './model_zoo/{}_checkpoint.pth'.format(args.ckpt))
                maximum_validation_psrn = validation_psrn
                print('new best model saved at epoch: {}'.format(epoch))
                print('-------------------------------------------------')
    print('-------------------------------------------------')
    print('best validation psnr achieved: {:.3f}'.format(maximum_validation_psrn))


def evaluate(model, validation_loader):
    psrn = []
    for data in validation_loader:
        noised_image = data['noised'].to(device)
        target = data['target'].to(device)
        noise, denoised_image = model(noised_image)
        psrn.append(calculate_psnr(target * 255, denoised_image * 255).item())
    return np.mean(psrn)


def resume(args, model, optimizer):
    checkpoint_path = './model_zoo/{}_checkpoint.pth'.format(args.ckpt)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    checkpoint_saved = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint_saved['model_state_dict'])
    optimizer.load_state_dict(checkpoint_saved['optimizer_state_dict'])

    print('Resume completed for the model\n')

    return model, optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='ircnn_color_image_denoiser_15')
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--test', type=int, default=0, help='test model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.resume = 1
    for noise_level in range(2, 51, 2):
        img_paths = ['C:/Users/Administrator/Desktop/adcv/Learning deep CNN denoiser prior for image restoration/raw_data/BSD/*',
                     'C:/Users/Administrator/Desktop/adcv/Learning deep CNN denoiser prior for image restoration/raw_data/ImageNet/*',
                     'C:/Users/Administrator/Desktop/adcv/Learning deep CNN denoiser prior for image restoration/raw_data/Waterloo/*'
                     ]
        out_put_path = '//data/train'
        process_raw_images(img_paths, out_put_path, noise_level)

        img_paths = ['C:/Users/Administrator/Desktop/adcv/Learning deep CNN denoiser prior for image restoration/data/test/CBSD68/*'
                     ]
        out_put_path = '//data/test/CBSD68_patches'

        process_raw_images(img_paths, out_put_path, noise_level)

        # args.test = 1
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        # dataloaders
        dataset = MyDataSet('./data/train')
        test_set = MyDataSet('./data/test/CBSD68_patches')
        trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        dataloaders = (trainloader, testloader)

        # network
        model = IRCNN(in_nc=3, out_nc=3).to(device)
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

        # resume the trained model
        if args.resume:
            model, optimizer = resume(args, model, optimizer)
        if args.test == 1:
            print('Average PSNR with noise level {n} : {p}'.format(n=noise_level, p=evaluate(model, testloader)))
        else:
            args.ckpt = 'ircnn_color_image_denoiser_{n}'.format(n=noise_level)
            train(args, model, optimizer, dataloaders)
            print('training finished')
