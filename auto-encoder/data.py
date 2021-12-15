import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from glob import glob
import PIL


class SceneDataset(Dataset):
    def __init__(self, scene_root, img_name = "final.png", transform=None):
        self.img_list = glob(os.path.join(scene_root, "*", img_name))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = PIL.Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image