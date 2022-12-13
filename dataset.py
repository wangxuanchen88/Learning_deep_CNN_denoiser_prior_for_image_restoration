import os
import PIL.Image as Image
import numpy as np
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataSet(Dataset):
    def __init__(self, dataset_path,grayscale = False):
        self.img_list = glob.glob(dataset_path+'/original/*')
        self.noised_img_list = glob.glob(dataset_path+'/noised/*')
        self.grayscale = grayscale


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        noised_img = Image.open(self.noised_img_list[idx])

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = transform(img)
        noised_img = transform(noised_img)
        if self.grayscale:
            img = transforms.Grayscale()(img)


        return {'noised': noised_img, 'target': img}
