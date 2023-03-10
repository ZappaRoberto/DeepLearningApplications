import torch.nn as nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, num_layer, in_channel, n_class):
        super().__init__()
        self.sequential = nn.Sequential()
        for n in range(num_layer - 1):
            self.sequential.append(nn.Linear(in_channel, in_channel))
            self.sequential.append(nn.ReLU())
            self.sequential.append(nn.Dropout(0.2))

        self.sequential.append(nn.Linear(in_channel, n_class))

    def forward(self, img):
        return self.sequential(img.flatten(1))
