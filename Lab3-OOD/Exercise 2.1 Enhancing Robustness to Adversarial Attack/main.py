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
import os
from PIL import Image


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
    epsilons = [0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1]  # Use smaller epsilon values
    accuracies = []
    examples = {eps: [] for eps in epsilons}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for eps in epsilons:
        correct = 0
        adv_examples = []

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            output = model(data)
            initial_pred = output.max(1, keepdim=True)[1]
            if initial_pred.item() != target.item():
                continue
            loss = torch.nn.functional.cross_entropy(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            # FGSM Attack without re-normalizing after perturbation
            perturbed_data = fgsm_attack(data, eps, data_grad)

            # Re-classify the perturbed image
            output = model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]

            if final_pred.item() == target.item():
                correct += 1
            if len(adv_examples) < 10:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((initial_pred.item(), final_pred.item(), adv_ex))


        final_acc = correct / float(len(test_loader))
        accuracies.append(final_acc)
        examples[eps] = adv_examples

        # Save the adversarial examples for the current epsilon
        for i, example in enumerate(adv_examples):
            # Save with consistent img_index
            plt.figure(figsize=(2, 2))
            plt.imshow(example[2].transpose(1, 2, 0))
            plt.axis('off')
            plt.savefig(f'examples/adv_{i}_eps_{eps}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

    # Plot accuracy vs epsilon
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, accuracies, '*-')
    plt.title("Epsilon vs Accuracy")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()


def create_collages_from_folder(folder_path):
    # Get all image file names from the folder
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

    # Dictionary to store images by index i
    images_by_index = {}

    # Organize images by their index 'i'
    for file in image_files:
        # Extract 'i' and 'eps' from the file name (assuming the format 'adv_{i}_eps_{eps}.png')
        base_name = file.split('_')
        i = base_name[1]  # Extract the index 'i'

        if i not in images_by_index:
            images_by_index[i] = []

        images_by_index[i].append(file)

    # Create a collage for each 'i'
    for i, files in images_by_index.items():
        # Sort files by epsilon (this ensures consistent order)
        files = sorted(files, key=lambda x: float(x.split('_eps_')[1].replace('.png', '')))

        # Load images
        pil_images = [Image.open(os.path.join(folder_path, f)) for f in files]

        # Determine the size of the collage (same height, width of all images combined)
        widths, heights = zip(*(img.size for img in pil_images))
        total_width = sum(widths)
        max_height = max(heights)

        # Create a new blank image with the combined width and max height
        collage = Image.new('RGB', (total_width, max_height))

        # Paste each image into the collage
        x_offset = 0
        for img in pil_images:
            collage.paste(img, (x_offset, 0))
            x_offset += img.width

        # Save the collage
        collage.save(os.path.join(folder_path, f'collage_{i}.png'))


if __name__ == '__main__':
    model = load_model()
    test_loader = load_cifar10_test()
    fgsm_attack_test(model, test_loader)
    create_collages_from_folder('examples')
