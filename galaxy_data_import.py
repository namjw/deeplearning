
import numpy as np
import matplotlib.pyplot as plt

import os

from PIL import Image

from glob import glob

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


img_path = glob('/home/jwnam/github/deeplearning/images_training_rev1/')[0]
label_path = glob('/home/jwnam/github/deeplearning/training_solutions_rev1.csv')[0]


class Galaxydataset(Dataset):
    
    def __init__(self, img_path=img_path, label_path=label_path, transform=None):
        # classes = ['Class1.1', 'Class1.2', ...]
        self.classes = np.loadtxt(label_path, dtype='str', delimiter=',',)[0][1:]
        
        # img_num = ['100008', ...]
        self.img_num = np.loadtxt(label_path, dtype='str', delimiter=',', skiprows=1, usecols=(0))
        
        # img_class = [[0.383147, 0.616853, ...], ...]
        self.img_class = np.loadtxt(label_path, dtype='float', delimiter=',', skiprows=1, ).T[1:].T
    
        self.tansform = transform
        
        self.img_list = [img_path+i+'.jpg' for i in img_num]
        
        
    def __len__(self):
        return len(self.img_num)
    
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        _img_filepath = self.img_list[idx]
        _img = Image.open(_img_filepath)
        
        _label = self.img_class[idx]
        
        if self.transform is not None:
            _img = self.transform(image=_img)["image"]
        
        return _img, _label

    
    
    
    