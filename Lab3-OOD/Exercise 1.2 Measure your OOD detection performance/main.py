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


def plot_roc(y_true, y_pred):
    plt.figure()
    ax = plt.gca()  # Get the current axes

    for i in range(y_true.shape[1]):
        RocCurveDisplay.from_predictions(y_true[:, i], y_pred[:, i], name=f'Class {i}', ax=ax)

    plt.title('ROC Curve for All Classes')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()

def plot_precision_recall(y_true, y_pred):
    plt.figure()
    ax = plt.gca()  # Get the current axes

    for i in range(y_true.shape[1]):
        PrecisionRecallDisplay.from_predictions(y_true[:, i], y_pred[:, i], name=f'Class {i}', ax=ax)

    plt.title('Precision-Recall Curve for All Classes')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.show()


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
        batch_size=2048,
        num_workers=4,
        pin_memory=False,
        shuffle=False,
        persistent_workers=True,
    )
    return test_loader


def load_model():
    model = CNN(in_channels=3, out_channels=64,
                n_class=10).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('../Exercise 1.1 Build a simple OOD detection pipeline/result/CNN-1/model.pth.tar')['state_dict'])
    model.eval()
    return model


def inference(model, test_loader):
    y_true = []
    y_pred = []
    for x, y in test_loader:
        x, y = x.to('cuda' if torch.cuda.is_available() else 'cpu'), y.to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            output = model(x)
            output = torch.nn.functional.softmax(output, dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(output.cpu().numpy())

    # Convert y_pred to a NumPy array
    y_pred = np.array(y_pred)

    # Binarize the y_true for multi-class ROC and Precision-Recall
    y_true = label_binarize(y_true, classes=range(10))

    plot_roc(y_true, y_pred)
    plot_precision_recall(y_true, y_pred)


if __name__ == '__main__':
    model = load_model()
    test_loader = load_cifar10_test()
    inference(model, test_loader)
