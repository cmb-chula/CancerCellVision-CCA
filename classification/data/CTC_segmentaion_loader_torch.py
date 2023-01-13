import time
import numpy as np
from .base_loader import BaseLoader
from sklearn.utils import shuffle
import numpy as np
import scipy
from scipy.stats import median_absolute_deviation
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
        self.train_transforms = A.Compose([
            
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            A.Rotate(),
            A.GaussianBlur(p=0.1),

            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
            # A.RandomGamma(),

            # A.Normalize((0.485, 0.456, 0.406, 0.485, 0.456, 0.406), (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
            A.Resize(128, 128)
        ])
        self.val_transforms = A.Compose([
            # A.Normalize((0.485, 0.456, 0.406, 0.485, 0.456, 0.406), (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
            A.Resize(128, 128)
        ])
        self.flag = flag
        self.cfg = cfg

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        #
        # add assert FL_only and read_RGU
        #

        img_path = self.data_path[idx]
        fl_path = '/'.join(img_path.split('/')[:5] + ['fluorescence'] + img_path.split('/')[6:])


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
        input_img = image.copy()

        if(self.cfg.read_U):
            input_img[2] = input_img[3]
            input_img = input_img[:3]
        elif(self.cfg.read_RGU):
            if(self.cfg.FL_only):
                input_img = input_img[3:]
        else:
            input_img = input_img[:3]
        # print(image.shape)
        input_img = torch.tensor(input_img, dtype = torch.float32)
        target_img = image[3:] / 255
        return input_img, target_img, label, img_path
