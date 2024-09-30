import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ImagesDataset(Dataset):
    def __init__(self, df_labels, root_dir, img_transform=None, mask_transform=None):
        self.df_labels = df_labels
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.df_labels)

    def __getitem__(self, idx):
        image_path = self.df_labels.iloc[idx]['image']
        mask_path = self.df_labels.iloc[idx]['mask']
        label = self.df_labels.iloc[idx]['label']
        
        image = cv2.imread(os.path.join(self.root_dir, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = torch.tensor(image).permute(2, 0, 1)

        mask = cv2.imread(os.path.join(self.root_dir, mask_path)) 
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.float32)[...,None] / 255.0
        mask = torch.tensor(mask).permute(2, 0, 1)

        label = torch.tensor(label.astype(np.float32)).unsqueeze(-1)

        if self.img_transform:
            image = self.img_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return {'image': image, 'mask': mask, 'label': label}