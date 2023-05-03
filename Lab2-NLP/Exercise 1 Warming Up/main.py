import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import albumentations as A
from model3 import UNet


WEIGHT_DIR = "result/FRT-1/model.pth.tar"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepareimage(img_path):
    img = Image.open(img_path).convert('RGB')
    image = np.array(img)

    aug = A.Compose([
        A.LongestMaxSize(max_size=512, interpolation=0, p=1),
        A.PadIfNeeded(min_height=344, min_width=344, p=1),
    ])
    augmented = aug(image=image)
    image = augmented['image']
    image = T.Compose([
        T.ToTensor(),
        T.Normalize((0.471, 0.448, 0.408), (0.234, 0.239, 0.242))])(image)
    image = image.unsqueeze(dim=0)
    _, _, h, w = image.shape
    img.resize([w, h]).show()
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
    main('Dataset/train/images/ff_00003.png')
