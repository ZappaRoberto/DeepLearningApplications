import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import albumentations as A
from model import UNet

WEIGHT_DIR = "result/v2.0/checkpoint.pth.tar"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepareimage(img_path):
    image = np.array(Image.open(img_path).convert('RGB'))
    aug = A.Compose([
        A.LongestMaxSize(max_size=1280, interpolation=0, p=1),
        A.PadIfNeeded(min_height=224, min_width=224, p=1),
    ])
    augmented = aug(image=image)
    image = augmented['image']
    image = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(image)
    image = image.unsqueeze(dim=0)
    print(image.shape)
    return image.to(DEVICE)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def main(img_path):
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load(WEIGHT_DIR), model)
    model.eval()
    image = prepareimage(img_path)
    prediction = model(image)
    prediction = (prediction > 0.5).float()
    prediction = prediction.squeeze(dim=0)
    example = T.ToPILImage()(prediction)
    example.show()


if __name__ == "__main__":
    main('Dataset/val/images/amz_00305.png')
