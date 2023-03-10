import torch.nn as nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, num_layer, in_channel, n_class):
        super().__init__()
        self.sequential = nn.Sequential()
        for n in num_layer:
            if n <= (num_layer - 1):
                self.sequential.append(nn.Linear(in_channel, in_channel))
                self.sequential.append(nn.ReLU())
            else:
                self.sequential.append(nn.Linear(in_channel, n_class))

    def forward(self, img):
        return self.sequential(img)
