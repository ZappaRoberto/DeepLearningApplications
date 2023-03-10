import os
from mnist import MNIST as mn
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class MNISTDataset(Dataset):
    def __init__(self, img_path, dataType):
        self.mndata = mn(img_path)
        if dataType == 'train':
            self.images, self.labels = self.mndata.load_training()
        else:
            self.images, self.labels = self.mndata.load_testing()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))])
        image = self.transform(np.array(self.images[0]))

        print(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        pass


def download_dataset():
    MNIST(root='./Dataset/', download=True)


if __name__ == "__main__":
    MNISTDataset('Dataset/MNIST/raw', 'val')
    # download_dataset()


