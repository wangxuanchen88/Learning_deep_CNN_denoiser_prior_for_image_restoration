import math

import numpy as np
import torch

import torch.nn as nn

nn.MSELoss()


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[2:]
    img1 = img1[:, :, border:h - border, border:w - border]
    img2 = img2[:, :, border:h - border, border:w - border]
    img1 = torch.clamp_(img1,min=0.0,max=255.0)
    img2 = torch.clamp_(img2,min=0.0,max=255.0)
    mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
    return torch.mean(20 * torch.log10(255.0 / torch.sqrt(mse)))
