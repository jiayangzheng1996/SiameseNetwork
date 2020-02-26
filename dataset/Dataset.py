from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset
import random
import numpy as np
import torch

class SiameseDataset(Dataset):
    def __init__(self, ImgDataset, transform=None, invert=True):
        self.ImgDataset = ImgDataset
        self.transform = transform
        self.invert = invert

    def __getitem__(self, index):
        img0_file = random.choice(self.ImgDataset.imgs)
        img1_file = random.choice(self.ImgDataset.imgs)
        if random.randint(0,1):
            while img0_file[1] is not img1_file[1]:
                img1_file = random.choice(self.ImgDataset.imgs)
        img0 = Image.open(img0_file[0])
        img1 = Image.open(img1_file[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img0_file[1] != img1_file[1])], dtype=np.float64))

    def __len__(self):
        return len(self.ImgDataset.imgs)
