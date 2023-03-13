<a href="https://pytorch.org/">
    <img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png" alt="Pytorch logo" title="Pytorch" align="right" height="80" />
</a>

# Deep Learning Applications

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Welcome to DeppLearningApplications!, a repository that contains multiple beginner exercises in deep learning! This repository is designed to provide you with a variety of exercises that cover different aspects of deep learning, including computer vision, natural language processing, deep reinforcement learning, and more.
Whether you are a beginner in deep learning or an experienced practitioner looking for a fun and challenging way to practice your skills, these exercises are designed to help you improve your understanding and develop your expertise in this exciting field.
Each exercise is accompanied by detailed instructions and solutions to help others learn and grow. So, let's get started and explore the fascinating world of deep learning together!
 


## Table Of Content

- [Laboratory 1 Convolutional Neural Networks](#Laboratory-1-Convolutional-Neural-Networks)
    - [Exercise 1: A baseline MLP](#Exercise-1:-A-baseline-MLP)
    - [Exercise 2: Rinse and Repeat](#Exercise-2:-Rinse-and-Repeat)
    - [Exercise 3: Why Residual Connections are so effective](#Exercise-3:-Why-Residual-Connections-are-so-effective)
    - [Exercise 4: Fully-convolutionalize a network](#Exercise-4:-Fully-convolutionalize-a-network)
    - [Exercise 5: Explain the predictions of a CNN](#Exercise-5:-Explain-the-predictions-of-a-CNN)
- [Dataset](#Dataset)
    - [Yahoo! Answer topic classification](#Yahoo!-Answer-topic-classification)
    - [Amazon Reviews](#Amazon-Reviews)
- [Training](#Training)
- [Result Analysis](#Result-Analysis)
    - [Text Classification](#Text-Classification)
    - [Sentiment Analysis](#Sentiment-Analysis)
- [How to use](#How-to-use)


## Laboratory 1: Convolutional Neural Networks

The overall architecture of this network is shown in the following figure:
<p align="center">
  <img src="https://github.com/ZappaRoberto/VDCNN/blob/main/img/architecture.png" />
</p>

The first block is a **`lookup table`** that generates a 2D tensor  of size (f0, s) that contain the embeddings of the s characters.

```python
class LookUpTable(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super(LookUpTable, self).__init__()
        self.embeddings = nn.Embedding(num_embedding, embedding_dim)

    def forward(self, x):
        return self.embeddings(x).transpose(1, 2)
```
> **Note**
> 
> The output dimension of the nn.Embedding layer is (s, f0). Use **`.transpose`** in order to have the right output dimension.

The second layer is a **`convolutional layer`** with in_channel dimension of 64 and kernel dimension of size 3.

```python
class FirstConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FirstConvLayer, self).__init__()
        self.sequential = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size))
```

The third layer is a **`convolutional block layer`** structured as shown in the following figure:
<p align="center">
  <img src="https://github.com/ZappaRoberto/VDCNN/blob/main/img/conv_block.png" />
</p>
We have also the possibility to add short-cut and in some layer we have to half the resolution with pooling. We can choose between three different pooling method: resnet like, VGG like or with k-max pooling.

```python
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, want_shortcut, downsample, last_layer, pool_type='vgg'):
        super(ConvolutionalBlock, self).__init__()

        self.want_shortcut = want_shortcut
        if self.want_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm1d(out_channels)
            )
```

with the variable **`want_shortcut`** we can choose if we want add shortcut to our net.

```python

        self.sequential = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
```

in this piece of code we build the core part of the convolutional block, as shown in the previously figure. self.conv1 can't be added in self.sequential because its stride depends on the type of pooling we want to use.

```python

        if downsample:
            if last_layer:
                self.want_shortcut = False
                self.sequential.append(nn.AdaptiveMaxPool1d(8))
            else:
                if pool_type == 'convolution':
                    self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                           kernel_size=3, stride=2, padding=1, bias=False)
                elif pool_type == 'kmax':
                    channels = [64, 128, 256, 512]
                    dimension = [511, 256, 128]
                    index = channels.index(in_channels)
                    self.sequential.append(nn.AdaptiveMaxPool1d(dimension[index]))
                else:
                    self.sequential.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.relu = nn.ReLU()
```

the final part of this layer manage the type of pooling that we want to use. We can select the pooling type with the variable **`pool_type`**. The last layer use always k-max pooling with dimension 8 and for this reason we manage this difference between previously layer with the variable **`last_layer`**.

```python
class FullyConnectedBlock(nn.Module):
    def __init__(self, n_class):
        super(FullyConnectedBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_class),
            nn.Softmax(dim=1)
        )
```

After the sequence of convolutional blocks we have 3 fully connected layer where we have to choose the output number of classes. Different task require different number of classes. We choose the number of classes with the variable **`n_class`**. Since we want to have the probability of each class given a text we use the softmax.

```python

class VDCNN(nn.Module):
    def __init__(self, depth, n_classes, want_shortcut=True, pool_type='VGG'):
```
The last class named VDCNN build all the layer in the right way and with the variable **`depth`** we can choose how many layer to add to our net. The paper present 4 different level of depth: 9, 17, 29, 49. You can find all theese piece of code inside the **model.py** file.

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>


## Dataset

The dataset used for the training part are the [Yahoo! Answers Topic Classification](https://www.kaggle.com/datasets/b78db332b73c8b0caf9bd02e2f390bdffc75460ea6aaaee90d9c4bd6af30cad2) and a subset of [Amazon review data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) that can be downloaded [here](https://drive.google.com/file/d/0Bz8a_Dbh9QhbZVhsUnRWRDhETzA/view?usp=share_link&resourcekey=0-Rp0ynafmZGZ5MflGmvwLGg). The vocabolary used is the same used in the paper: **"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%^&*~‘+=<>()[]{} "**. I choose to use 0 as a value for padding and 69 as a value for unknown token. All this datasets are maneged by **`Dataset class`** inside dataset.py file. 


### Yahoo! Answer topic classification

The Yahoo! Answers topic classification dataset is constructed using the 10 largest main categories. Each class contains 140000 training samples and 6000 testing samples. Therefore, the total number of training samples is 1400000, and testing samples are 60000. The categories are:

* Society & Culture
* Science & Mathematics
* Health
* Education & Reference
* Computers & Internet
* Sports
* Business & Finance
* Entertainment & Music
* Family & Relationships
* Politics & Government


### Amazon Reviews

The Amazon Reviews dataset is constructed using 5 categories (star ratings).

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## Training

> **Warning**
> 
> Even if it can be choosen the device between cpu or GPU, I used and tested the training part only with GPU.

First things first, at the beginning of train.py file there are a some useful global variable that manage the key settings of the training.

```python

LEARNING_RATE = 0.01
MOMENTUM = 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
MAX_LENGTH = 1024
NUM_EPOCHS = 1
PATIENCE = 40
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = "dataset/amazon/train.csv"
TEST_DIR = "dataset/amazon/test.csv"
```

> **Note**
> 
> Change **`TRAIN_DIR`** and **`TEST_DIR`** with your datasets local position.

The train_fn function is build to run one epoch and return the average loss and accuracy of the epoch.

```python

def train_fn(epoch, loader, model, optimizer, loss_fn, scaler):
    # a bunch of code
    return train_loss, train_accuracy
```

The main function is build to inizialize and manage the training part until the end.

```python

def main():
    model = VDCNN(depth=9, n_classes=5, want_shortcut=True, pool_type='vgg').to(DEVICE)
    # training settings
    for epoch in range(NUM_EPOCHS):
        # run 1 epoch
        # check accuracy
        # save model
        # manage patience for early stopping
    # save plot
    sys.exit()
```

> **Note**
> 
> Remember to change **`n_classes`** from 5 to 10 if you use Amazon dataset or Yahoo! Answer dataset.

**`get_loaders`**, **`save_checkpoint`**, **`load_checkpoint`**, **`check_accuracy`** and **`save_plot`**  are function used inside tran.py that can be finded inside utils.py.

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## Result Analysis

For computational limitation I trained the models only with depth 9. the result showed below are the test error of my implementation and paper implementation.

### Text Classification

|     Pool Type     |  My Result  | Paper Result |
| :---------------: | :---------: | :----------: |
| Convolution       |    32.57    |     28.10    |
| KMaxPooling       |    28.92    |     28.24    |
| MaxPooling        |    28.40    |     27.60    |


### Sentiment Analysis

|     Pool Type     |  My Result  | Paper Result |
| :---------------: | :---------: | :----------: |
| Convolution       |    40.35    |     38.52    |
| KMaxPooling       |    38.58    |     39.19    |
| MaxPooling        |    38.45    |     37.95    |

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## How to use

After training run main.py file changing variable **`WEIGHT_DIR`** with the local directory where the weight are saved

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>
