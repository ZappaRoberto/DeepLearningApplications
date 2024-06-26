<a href="https://pytorch.org/">
    <img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png" alt="Pytorch logo" title="Pytorch" align="right" height="80" />
</a>

# Deep Learning Applications

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Welcome to DeppLearningApplications!, a repository that contains multiple beginner exercises in deep learning! This repository is designed to provide you with a variety of exercises that cover different aspects of deep learning, including computer vision, natural language processing, deep reinforcement learning, and more.
Whether you are a beginner in deep learning or an experienced practitioner looking for a fun and challenging way to practice your skills, these exercises are designed to help you improve your understanding and develop your expertise in this exciting field.
Each exercise is accompanied by detailed instructions and solutions to help others learn and grow. So, let's get started and explore the fascinating world of deep learning together!
 


## Table Of Content

- [Laboratory 1: Convolutional Neural Networks](#Laboratory-1-Convolutional-Neural-Networks)
    - [Exercise 1: A baseline MLP](#Exercise-1-A-baseline-MLP)
    - [Exercise 2: Rinse and Repeat](#Exercise-2-Rinse-and-Repeat)
    - [Exercise 3: Why Residual Connections are so effective](#Exercise-3-Why-Residual-Connections-are-so-effective)
    <!---
  - [Exercise 4: Fully-convolutionalize a network](#Exercise-4-Fully-convolutionalize-a-network)
    - [Exercise 5: Explain the predictions of a CNN](#Exercise-5-Explain-the-predictions-of-a-CNN)
    - [Exercise 6: Instance segmentation?](#Exercise-5-Explain-the-predictions-of-a-CNN)
    - [Exercise 7: Siamese models?](#Exercise-5-Explain-the-predictions-of-a-CNN)
  -->
- [Laboratory 2: Natural Language Processing](#Laboratory-2-Natural-Language-Processing)
    - [Exercise 1: Warming Up](#Exercise-1-Warming-Up)
    - [Exercise 2: Working with Real LLMs](#Exercise-2-Working-with-Real-LLMs)
    - [Exercise 3: Reusing Pre-trained LLMs](#Exercise-3-Reusing-Pre-trained-LLMs)
- [Laboratory 3: Adversarial Learning and OOD Detection](#Laboratory-3-Adversarial-Learning-and-OOD-Detection)
    - [Exercise 1: A baseline MLP](#Exercise-1-A-baseline-MLP)
    - [Exercise 1: A baseline MLP](#Exercise-1-A-baseline-MLP)
    - [Exercise 1: A baseline MLP](#Exercise-1-A-baseline-MLP)
    - [Exercise 1: A baseline MLP](#Exercise-1-A-baseline-MLP)


## Laboratory 1: Convolutional Neural Networks

In this first laboratory you will get some practice working with Deep Models in (somewhat) sophisticated ways. We will reproduce (on a small scale) the results of the Residual Networks paper, demonstrating that deeper does not always mean better. Subsequent exercises will ask you to delve deeper into the inner workings of CNNs.
Try to implement your own training pipeline and use my code only as inspiration.
> **Note**
> 
> This would be a good time to think about abstracting your model definition, and training and evaluation pipelines in order to make it easier to compare performance of different models.
> This would also be a great point to study how Tensorboard or Weights and Biases can be used for performance monitoring.
> In my code I will use Weights and Biases.


## Exercise 1: A baseline MLP

Implement a simple Multilayer Perceptron to classify the 10 digits of MNIST (e.g. two narrow layers). Train this model to convergence, monitoring (at least) the loss and accuracy on the training and validation sets for every epoch.


### Architecture:

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise1/mlp.png" />
</p>

### Result:

|     Net     |  Accuracy  |  Loss  |
| :---------: | :--------: |  :--:  |
| MLP         |    32.57   |  28.10 |

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>


## Exercise 2: Rinse and Repeat

Repeat the verification of exercise 1, but with **Convolutional** Neural Networks. Show that **deeper** CNNs *without* residual connections do not always work better and **even deeper** ones *with* residual connections. For this exercise I will use CIFAR10, since MNIST is *very* easy. I choose to run at most 200 epochs with a patience of 20 epochs.


### Introduction:

Since the introduction of ResNet in 2015, residual connections (also known as skip connections) have become one of the most important architectural decisions in deep learning, enabling the construction of deeper and better-performing models. Residual connections are a way to address the problem of vanishing gradients. However, the use of skip connections is not always better and may depend on the specific task, dataset and model. Therefore, the question of whether or not to use residual connections in a neural network and how deep to make the network remains an important area of research in deep learning.


### Architecture:

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise2/CNN.png" />
</p>

### Result:

First of all I choose a model with 9 Convolutional Layer as a baseline for this exercise. After being trained the model achieve a 88.74% of accuracy as shown in the followinf Figure: <br>

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise2/baseline.png" />
</p>

Now, what's happen if I train on the same dataset deeper models with 17 or even 48 layers? What's happen if I add residual connections to all this models? <br> 
Let's figure out! <br>
First things first, are deeper model always better than shallower ones? <br>
As you can see in the following figure, the model with depth 17 is marginally better than the one with depth 9. However, this is not true in the case of depth 48 where the performance is worse. <br>

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise2/models.png" />
</p>


What's happen if I add residual connections to all this models? <br>
First of all I want to analaze what's happen with residual connection for depth 9 and 17: <br>
As show in the following figure adding residual connection to models that are not enought deeper worsens the results.

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise2/skip.png" />
</p>

and the deepr model? <br>
As shown in the following figure not even the deeper model with residual connection can achieve better result than the shallower ones without skip connection 

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise2/best.png" />
</p>

### Table Result:


The following table summarize all the previously results:

|      Net      |  Accuracy  |   Loss   |
| :-----------: | :--------: | :------: |
| Depth-9       |    88.74   |  0.5249  |
| Depth-9-skip  |    88.20   |  0.5433  |
| **Depth-17**      |    **89.65**   |  **0.4871**  |
| Depth-17-skip |    89.51   |  0.5189   | 
| Depth-48      |    86.85   |  0.5053   |
| Depth-48-skip |    87.96   |  0.5245   |

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>


## Exercise 3: Why Residual Connections are so effective


Use your two models (with and without residual connections) you developed before to study and quantify why the residual versions of the networks learn more effectively.<br>

### Introduction

The vanishing gradient problem refers to the issue where the gradients of the loss function with respect to the weights of a deep neural network become very small during backpropagation. This can lead to very slow learning or complete failure of the neural network to converge. <br>
The problem occurs when the weights of the network are updated using the chain rule of differentiation during backpropagation. In deep neural networks, there are often many layers between the input and output layers, and the gradients are multiplied at each layer. If the gradients are small, they become increasingly smaller as they propagate backward through the layers, and can effectively "vanish" as they approach the input layer. <br>
To demonstrate the vanishing gradient problem, I will experiment with 48 deep models, both with and without skip connections. During training, I will closely monitor the weights of the very first layer, where the vanishing gradient problem is more pronounced, and then compare the results.


### Result

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise3/48s.png" />
</p>

Watch the following pictures, which one refer to the models with skip connection? Why?

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise3/gradient.png" />
</p>

On the right, we have the very first layer from the model with skip connections, while on the left, we have the very first layer from the model without skip connections. As you can see, there is a significant difference between these two versions. At the beginning of training, the weights on the right range from 8000 to -6000 and at the end from 2000 to -2000. In contrast, on the left, the weights start at around 200 and end up only slightly larger, representing more or less an order of magnitude difference!. <br>
What's happen, instead on the last convolution of the network?

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise3/lastlayer.png" />
</p>

As you can see the difference between the model with and without skip connection persist but we can also see that the gradient is exponentially increasing going backward, is this the famous problem of "Exploding gradient?"

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise3/48clipped.png" />
</p>

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>



## Laboratory 2: Natural Language Processing
In this laboratory we will get our hands dirty working with Large Language Models (e.g. GPT and BERT) to do various useful things.

## Exercise 1: Warming Up
In this first exercise you will train a small autoregressive GPT model for character generation (the one used by Karpathy in his video) to generate text in the style of Dante Aligheri. Use this file, which contains the entire text of Dante’s Inferno (note: you will have to delete some introductory text at the top of the file before training). Train the model for a few epochs, monitor the loss, and generate some text at the end of training. Qualitatively evaluate the results.
For this exercise I decided to implement from scratch LLama2 (model.py) and the training script. I did this model ~8 months ago. Now could be beautiful improve the architecture with the latest advancement (MoE, MoD) and also improve the training script but I don't know if I'll have the time to do it :( .
I made three major experiment on my RTX 4090. The model have ~15M parameters with a fixed vocab_size of 5549 and an embedding_size of 768. The contex lengh is 128 token. Two experiments will use 16 Heads, the other 4 heads. 

|      experiments      |      n_head      |  Test Loss  |   Epochs   |
| :-----------: | :-----------: | :--------: | :------: |
| 0       | 4       |    5.97   |  144  |
| 2       | 16  |    5.827   |  62  |
| 1       | 16      |    5.832   |  40  |

With the experiments 0 I prove that my implementation can learn as expected from the datasets while with the other two experiments I test how the number of heads are correlated with the convergence speed.

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise4/train.png" />
</p>
Looking the test loss we can easily see how quickly this architecture overfit.
<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise4/test.png" />
</p>

## Exercise 2: Working with Real LLMs
In this exercise we will see how to use the [Hugging Face](https://huggingface.co/) model and dataset ecosystem to access a *huge* variety of pre-trained transformer models.<br>
Instantiate the GPT2Tokenizer and experiment with encoding text into integer tokens.<br>
```python
def main(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    encoded_input = tokenizer(text, return_tensors="pt")

    # Compare the length of input text with the encoded sequence length
    input_length = len(text.split())
    encoded_length = encoded_input["input_ids"].shape[1]

    print(f"Original text length: {input_length}")
    print(f"Encoded sequence length: {encoded_length}")

    decoded_text = tokenizer.decode(encoded_input["input_ids"][0])
    print(f"Decoded text: {decoded_text}")
```
```python
text = "This is an example sentence to encode into tokens."
main(text)
```
The output of this experiments is:
- Original text length: 9
- Encoded sequence length: 10
- Decoded text: This is an example sentence to encode into tokens.<br>

Instantiate a pre-trained GPT2LMHeadModel and use the generate() method to generate text from a prompt.<br>
```python
def generate_text(prompt, model_name="gpt2", max_length=50, num_return_sequences=1, do_sample=True, top_k=50,
                  temperature=0.7):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    torch.manual_seed(42)  # Set the seed for reproducibility

    generated_text = model.generate(
        input_ids=tokenizer.encode(prompt, return_tensors="pt"),
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        top_k=top_k,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return generated_text
```
```python
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)
```
The output of this experiments is:<br>
Once upon a time, what was the best way to go about it? When I first started out, I was happy to do some of the things I loved about cooking: I didn't have to cook your food. I could<br>



## Exercise 3.1: Training a Text Classifier

The overall architecture of this network is shown in the following figure:
<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/prova.png" />
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
