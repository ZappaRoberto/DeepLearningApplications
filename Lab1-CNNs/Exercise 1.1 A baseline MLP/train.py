import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import MultilayerPerceptron
from utils import (
    get_loaders,
    save_checkpoint,
    load_checkpoint,
    check_accuracy,
    save_plot,
    Lion)

# Hyperparameters and other settings
LEARNING_RATE = 1e-6  # 1e-5
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
NUM_EPOCHS = 2000
PATIENCE = 20
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = 'Dataset/train/images'  # "COCOdataset2017/annotations/instances_train2017.json"
TEST_DIR = 'Dataset/val/images'  # "COCOdataset2017/annotations/instances_val2017.json"
WEIGHT_DIR = "result/v2.0/checkpoint.pth.tar"


def train_fn(epoch, loader, model, optimizer, scheduler, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    loop.set_description(f"Epoch {epoch}", refresh=True)

    running_loss = 0
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)

        # forward
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, target)

        # backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item()

        # dice score
        prediction = (prediction > 0.5).float()
        # num_correct += (prediction == target).sum()
        # num_pixels += torch.numel(prediction)
        dice_score += ((2 * (prediction * target).sum()) / ((prediction + target).sum() + 1e-8)).detach().cpu()

    loop.close()
    train_loss = running_loss / len(loop)
    # train_accuracy = num_correct/num_pixels*100
    train_dice_score = dice_score / len(loader)

    return train_loss, train_dice_score


def main():
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    if LOAD_MODEL:
        load_checkpoint(torch.load(WEIGHT_DIR), model)
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    optimizer = Lion(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, test_loader = get_loaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=1e-4,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=NUM_EPOCHS)
    train_l, train_d, test_l, test_d = [], [], [], []
    patience = PATIENCE
    max_test_dice = 0
    for epoch in range(NUM_EPOCHS):
        # train_loss, train_dice = train_fn(epoch, train_loader, model, optimizer, scheduler, loss_fn, scaler)
        # check accuracy
        test_loss, test_dice = check_accuracy(test_loader, model, loss_fn, device=DEVICE)
        # train_l.append(train_loss)
        # train_d.append(train_dice)
        test_l.append(test_loss)
        test_d.append(test_dice)
        # save model
        if test_dice > max_test_dice:
            max_test_dice = test_dice
            patience = PATIENCE
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            #save_checkpoint(checkpoint)
        if patience == 0:
            break
        patience -= 1
    save_plot(train_l, train_d, test_l, test_d)
    sys.exit()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
