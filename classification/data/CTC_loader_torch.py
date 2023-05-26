import time
import numpy as np
from .base_loader import BaseLoader
from sklearn.utils import shuffle
import numpy as np
import scipy
import pickle
from scipy.ndimage import rotate
import torchvision.transforms as T
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import albumentations as A
import cv2

import os
import pandas as pd
from PIL import Image
class CTCLoader(Dataset):
    def __init__(self, data_path, cfg, flag = 'train'):
        self.data_path = data_path
        self.flag = flag
        self.cfg = cfg

        self.train_transforms = A.Compose([
            
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            A.Rotate(),
            A.GaussianBlur(p=0.1),

            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
            # A.RandomGamma(),
            # A.Normalize((14.36845, 14.36845, 14.36845, 0.6833783, 2.8144684, 1.9409757), (3.7082622, 3.7082622, 3.7082622, 1.3245853, 3.2033563, 3.1374023)),
            A.Resize(self.cfg.img_size[0], self.cfg.img_size[1])
        ])
        self.val_transforms = A.Compose([
            # A.Normalize((14.36845, 14.36845, 14.36845, 0.6833783, 2.8144684, 1.9409757), (3.7082622, 3.7082622, 3.7082622, 1.3245853, 3.2033563, 3.1374023)),
            A.Resize(self.cfg.img_size[0], self.cfg.img_size[1])
        ])



    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        #
        # add assert FL_only and read_RGU
        #


        img_path = self.data_path[idx]
        fl_path = img_path.replace("brightfield", "fluorescence")

        image = cv2.imread(img_path)
        fl_image = cv2.imread(fl_path)
        image = np.array(np.concatenate([image, fl_image], axis = 2), dtype = np.float32) # [G, G, G, B, G, R] image channel
        class_idx = None
        classes = img_path.split('/')[-2]
        if(self.flag == 'train'):
            image = self.train_transforms(image=image)['image']
            label = self.cfg.class_mapper[classes]
        else:
            image = self.val_transforms(image=image)['image']
            label = self.cfg.test_class_mapper[classes]
        image = np.transpose(image, (2, 0, 1))


        if(self.cfg.read_U):
            image[2] = image[3]
            image = image[:3]
        elif(self.cfg.read_RGU):
            if(self.cfg.FL_only):
                image = image[3:]
        else:
            image = image[:3]
        # print(image.shape)
        image = torch.tensor(image, dtype = torch.float32)
        return image, label, img_path
