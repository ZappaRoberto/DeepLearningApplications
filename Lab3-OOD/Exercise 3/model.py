import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, n_class):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels,
                      kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(out_channels * 36, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_class),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


if __name__ == '__main__':
    model = CNN(3, 64, 10)
    print(model)
    print('=> Model built successfully !')
    print('=> Model architecture: CNN')
    print('=> Model parameters: in_channels=3, out_channels=64, n_class=10')
    print('=> Model output: 10 classes')
    print('=> Model input: 3 channels')
    print('=> Model output: 64 channels')
