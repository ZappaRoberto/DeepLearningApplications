import os
import torch
from matplotlib import pyplot as plt
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import skimage
from PIL import Image
import torchvision.transforms as T
from skimage.transform import resize
import torchvision.transforms.functional as F
import h5py
import albumentations as A
import albumentations.pytorch.transforms as AT
import random

# TODO: hdf5 of the images for speed up loading
# TODO: optimize what it can be optimized


class CustomDataset(Dataset):
    def __init__(self, img_path, dataType):
        self.list = os.listdir(img_path)
        self.dataType = dataType
        self.aug = A.Compose([
            A.LongestMaxSize(max_size=224, interpolation=0, p=1),
            A.PadIfNeeded(min_height=224, min_width=224, p=1),
            A.OneOf([
                A.CoarseDropout(max_holes=random.randint(1, 15), max_height=random.randint(1, 8),
                                max_width=random.randint(1, 8), p=0.5),
                A.MaskDropout(p=0.5),
                # A.GridDropout(p=0.5),
                A.RandomShadow(p=0.5)], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Perspective(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.CLAHE(p=0.5),
            A.FancyPCA(p=0.5),
            A.OneOf([
                # A.ColorJitter(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                # A.HueSaturationValue(p=0.5),
                # A.RandomGamma(p=0.5),
                # A.Solarize(p=0.5)
            ], p=0.5),
            A.OneOf([
                A.ISONoise(),
                A.GaussNoise(),
                A.MultiplicativeNoise()
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.5),
                A.MedianBlur(p=0.5),
                A.GlassBlur(p=0.5),
                A.GaussianBlur(p=0.5),
                A.Blur(p=0.5)], p=0.5),
            A.Sharpen(p=0.5),
            A.OneOf([
                A.RandomFog(p=0.5),
                A.RandomRain(p=0.5),
            ], p=0.5),
        ], p=1)
        self.val = A.Compose([
            A.LongestMaxSize(max_size=224, interpolation=0, p=1),
            A.PadIfNeeded(min_height=224, min_width=224, p=1),
        ])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        image = np.array(Image.open('Dataset/{}/images/{}'.format(self.dataType, self.list[idx])).convert('RGB'))
        mask = np.array(Image.open('Dataset/{}/segmask/{}'.format(self.dataType, self.list[idx])))

        if self.dataType == 'val':
            augmented = self.val(image=image, mask=mask)
        else:
            augmented = self.aug(image=image, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        mask = torch.from_numpy(mask).unsqueeze(dim=0).float()
        image = self.normalize(image)
        return image, mask


class COCODataset(Dataset):
    def __init__(self, img_path, dataType):
        self.coco = COCO(img_path)
        self.catIDs = self.coco.getCatIds()
        self.imgIds = self.coco.getImgIds()
        self.dataType = dataType
        # self.file = h5py.File('COCOdataset2017/images/{}'.format(self.dataType), 'r')
        self.aug = A.Compose([
            A.LongestMaxSize(max_size=224, interpolation=0, p=1),
            A.PadIfNeeded(min_height=224, min_width=224, p=1),
            A.OneOf([
                A.CoarseDropout(max_holes=random.randint(1, 15), max_height=random.randint(1, 8),
                                max_width=random.randint(1, 8), p=0.5),
                A.MaskDropout(p=0.5),
                # A.GridDropout(p=0.5),
                A.RandomShadow(p=0.5)], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Perspective(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.CLAHE(p=0.5),
            A.FancyPCA(p=0.5),
            A.OneOf([
                # A.ColorJitter(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                # A.HueSaturationValue(p=0.5),
                # A.RandomGamma(p=0.5),
                # A.Solarize(p=0.5)
            ], p=0.5),
            A.OneOf([
                A.ISONoise(),
                A.GaussNoise(),
                A.MultiplicativeNoise()
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.5),
                A.MedianBlur(p=0.5),
                A.GlassBlur(p=0.5),
                A.GaussianBlur(p=0.5),
                A.Blur(p=0.5)], p=0.5),
            A.Sharpen(p=0.5),
            A.OneOf([
                A.RandomFog(p=0.5),
                A.RandomRain(p=0.5),
            ], p=0.5)
        ], p=1)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.val = A.Compose([
            A.LongestMaxSize(max_size=224, interpolation=0, p=1),
            A.PadIfNeeded(min_height=224, min_width=224, p=1),
        ])

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        img = self.coco.loadImgs(self.imgIds[idx])[0]
        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIDs, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        mask = np.zeros((img['height'], img['width']))
        for i in range(len(anns)):
            mask = np.maximum(self.coco.annToMask(anns[i]), mask)
        # input = self.file['images'][index, :, :]
        I = np.array(Image.open('COCOdataset2017/images/{}/{}'.format(self.dataType, img['file_name'])).convert('RGB'))

        if self.dataType == 'val':
            augmented = self.val(image=I, mask=mask)
        else:
            augmented = self.aug(image=I, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        mask = torch.from_numpy(mask).unsqueeze(dim=0)
        image = self.transform(image)
        # example = T.ToPILImage()(mask)
        # example.show()
        return image, mask


def find(img_path, dataType):
    lists = os.listdir(img_path)
    for idx in lists:
        mask = np.array(Image.open('Dataset/{}/segmask/{}'.format(dataType, idx)))
        mask[mask == 255] = 1
        example = T.ToPILImage()(mask)
        example.save('Dataset/{}/segmask/{}'.format(dataType, idx))



if __name__ == "__main__":
    # find('Dataset/val/segmask', 'val')
    CustomDataset('Dataset/val/segmask', 'val')
    # COCODataset("COCOdataset2017/annotations/instances_val2017.json", 'val')

