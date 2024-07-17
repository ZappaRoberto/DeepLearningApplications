import numpy as np
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from model import CNN
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay


def load_cifar10_test():
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    test_ds = datasets.CIFAR10(root='../Exercise 1.1 Build a simple OOD detection pipeline/Dataset/',
                               train=False,
                               download=False,
                               transform=transform_test)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=4,
        pin_memory=False,
        shuffle=False,
        persistent_workers=True,
    )
    return test_loader


def load_model():
    model = CNN(in_channels=3, out_channels=64,
                n_class=10).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('../Exercise 1.1 Build a simple OOD detection pipeline/result/CNN-2/model.pth.tar')['state_dict'])
    model.eval()
    return model


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def denorm(batch, mean=[0.5], std=[0.5]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def fgsm_attack_test(model, test_loader):
    # Test adversarial attack with different epsilon values from 0 to 0.1
    epsilons = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    accuracies = []
    examples = {eps: [] for eps in epsilons}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loop over all epsilon values
    for eps in epsilons:
        correct = 0
        adv_examples = []

        # Loop over all examples in test set
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
            # Forward pass the data through the model
            output = model(data)
            # If the initial prediction is wrong, don't bother attacking, just move on
            initial_pred = output.max(1, keepdim=True)[1]
            if initial_pred.item() != target.item():
                continue
            # Calculate the loss
            loss = torch.nn.functional.cross_entropy(output, target)
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrad
            data_grad = data.grad.data
            data = denorm(data)
            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, eps, data_grad)
            perturbed_data = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(perturbed_data)
            # Re-classify the perturbed image
            output = model(perturbed_data)
            # Check for success
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                correct += 1
                # Save some examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((initial_pred.item(), final_pred.item(), adv_ex))

        # Calculate final accuracy for this epsilon
        final_acc = correct / float(len(test_loader))
        accuracies.append(final_acc)
        examples[eps] = adv_examples

    # Plot accuracy vs epsilon
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, accuracies, '*-')
    plt.title("Epsilon vs Accuracy")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()
    # save in the example folder the adversarial examples
    for eps, adv_examples in examples.items():
        for i, example in enumerate(adv_examples):
            plt.figure(figsize=(2, 2))
            plt.imshow(example[2].transpose(1, 2, 0))
            plt.axis('off')
            plt.savefig(f'examples/adv_{i}_eps_{eps}.png', bbox_inches='tight', pad_inches=0)
            plt.close()


if __name__ == '__main__':
    model = load_model()
    test_loader = load_cifar10_test()
    fgsm_attack_test(model, test_loader)