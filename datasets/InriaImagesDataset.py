import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ImagesDataset(Dataset):
    def __init__(self, df_labels, root_dir, transform=None):
        self.df_labels = df_labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df_labels)

    def __getitem__(self, idx):
        image_path = self.df_labels.iloc[idx]['image']
        mask_path = self.df_labels.iloc[idx]['mask']
        
        image = cv2.imread(os.path.join(self.root_dir, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.root_dir, mask_path))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return {'image': image, 'mask': mask.unsqueeze(0)}