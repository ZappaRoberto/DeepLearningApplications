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
    - [Exercise 1: OOD Detection and Performance Evaluation](#Exercise-1-OOD-Detection-and-Performance-Evaluation)
    - [Exercise 2: Enhancing Robustness to Adversarial Attack](#Exercise-2-Enhancing-Robustness-to-Adversarial-Attack)
    - [Exercise 3: Wildcard](#Exercise-3-Wildcard)


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



## Exercise 3: Reusing Pre-trained LLMs

For this exercise I will finetune DistilRoberta-base for a topic classification task.

## Dataset

The dataset used for the training part are the [Yahoo! Answers Topic Classification](https://www.kaggle.com/datasets/b78db332b73c8b0caf9bd02e2f390bdffc75460ea6aaaee90d9c4bd6af30cad2)


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


## Result Analysis

|     architecture     |  Accuracy  |
| :---------------: | :---------: |
| DistilRoberta-base       |    75.8    |

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise5/test.png" />
</p>
<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise5/train.png" />
</p>
<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise5/loss.png" />
</p>


<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## Laboratory 3: Adversarial Learning and OOD Detection
In this laboratory session we will develop a methodology for detecting OOD samples and measuring the quality of OOD detection. We will also experiment with incorporating adversarial examples during training to render models more robust to adversarial attacks.

## Exercise 1: OOD Detection and Performance Evaluation
In this first exercise you will build a simple OOD detection pipeline and implement some performance metrics to evaluate its performance. Your *OOD Detector* should produce a score representing how "out of distribution" a test sample is. For this exercise I choose to calculate the id score and the ood score as 1 - max_softmax_scores. In this way Higher score indicates more "out of distribution"

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise6/metrics.png" />
</p>

AS you can see Is clear that during training the id score tend to move to zero while the ood score even if descrease as well tend to stationate around ~4k.<br>
There are several metrics used to evaluate OOD detection performance, we will concentrate on two threshold-free approaches: the area under the Receiver Operator Characteristic (ROC) curve for ID classification, and the area under the Precision-Recall curve for *both* ID and OOD scoring.

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise6/ROC.png" />
</p>
<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise6/recallprecision.png" />
</p>


<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>


## Exercise 2: Enhancing Robustness to Adversarial Attack

In this second exercise we will experiment with enhancing our base model to be (more) robust to adversarial attacksusing the Fast Gradient Sign Method (FGSM) that perturbs samples in the direction of the gradient with respect to the input

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise7/myplot.png" />
</p>

I show an example of perturbation<br>

<p align="left">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise7/adv_3_eps_0.png" />
</p>

<p align="right">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise7/adv_3_eps_0.09.png" />
</p>

I Used my implementation of FGSM to augment *on the fly* the training dataset with adversarial samples. In this way the adversarial samples are always generated using the current model.

<p align="center">
  <img src="https://github.com/ZappaRoberto/DeepLearningApplications/blob/main/img/exercise7/myplot-2.2.png" />
</p>

The model as you can see is more robust to OOD

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>


## Exercise 3: Wildcard

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>
