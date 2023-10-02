from torch.utils.data import Dataset
import torch
import cv2
from PIL import Image
import os

class FlowerDataset(Dataset):
    def __init__(self, path, classes):
        self.path = path
        self.classes= classes
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.path[idx])
        img_tensor = torch.from_numpy(img)
        class_id = torch.tensor([self.classes[idx]])
        return img_tensor, class_id



