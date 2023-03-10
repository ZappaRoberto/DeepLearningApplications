import torch
from torch.utils.data import DataLoader
from dataset import COCODataset, CustomDataset
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as T


def save_checkpoint(state, filename="result/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_dir, test_dir, batch_size, num_workers, pin_memory):

    train_ds = CustomDataset(
        img_path=train_dir,
        dataType='train'
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_ds = CustomDataset(
        img_path=test_dir,
        dataType='val'
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, test_loader


def check_accuracy(loader, model, loss_fn, device):
    running_loss = 0
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = torch.sigmoid(model(data))
            loss = loss_fn(prediction, target)
            running_loss += loss.item()
            prediction = (prediction > 0.5).float()
            num_correct += (prediction == target).sum()
            num_pixels += torch.numel(prediction)
            dice_score += ((2 * (prediction * target).sum()) / ((prediction + target).sum() + 1e-8)).detach().cpu()

    loss = running_loss / len(loader)
    accuracy = num_correct/num_pixels*100
    dice_score = dice_score/len(loader)
    print(f"Got on test set --> Dice score: {dice_score:.6f}, Accuracy: {accuracy:.3f} and Loss: {loss:.3f}")
    model.train()
    return loss, dice_score


def save_plot(train_l, train_a, test_l, test_a):
    plt.plot(train_a, '-')
    plt.plot(test_a, '-')
    plt.xlabel('epoch')
    plt.ylabel('dice score')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Dice Score')
    plt.savefig('result/dice')
    plt.close()

    plt.plot(train_l, '-')
    plt.plot(test_l, '-')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig('result/losses')
    plt.close()


class Lion(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss