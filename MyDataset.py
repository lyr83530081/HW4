# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:41:48 2021

@author: user
"""

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform

        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)
    
# train_data=MyDataset(txt='train.txt', transform=transforms.ToTensor())
# aaa,bbb = train_data.__getitem__(1)
# aaaa=aaa.numpy()
# data_loader = DataLoader(train_data, batch_size=100,shuffle=True)
# print(len(data_loader))