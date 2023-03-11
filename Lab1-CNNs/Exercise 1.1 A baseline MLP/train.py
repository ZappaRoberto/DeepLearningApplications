import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from model import MultilayerPerceptron
from utils import (get_loaders, save_checkpoint, load_checkpoint, metrics, eval_fn, save_plot, Lion)
import wandb

# Hyperparameters and other settings
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16384
NUM_EPOCHS = 5
PATIENCE = 20
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = False
WEIGHT_DIR = "result/checkpoint.pth.tar"


# TODO: add Weight and bias


def train_fn(epoch, loader, model, optimizer, scheduler, loss_fn, scaler, metric_collection):
    model.train()
    running_loss = 0

    for (data, target) in tqdm(loader, desc=f"Epoch {epoch + 1}"):
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
        metric_collection(prediction, target)

    train_loss = running_loss / len(loader)
    train_accuracy = metric_collection['MulticlassAccuracy'].compute().cpu() * 100

    metric_collection.reset()

    return train_loss, train_accuracy


def main():
    prova = wandb.init(
        # set the wandb project where this run will be logged
        project="MultilayerPerceptron",
        group='Multi Layer Perceptron',
        tags=["baseline"],
        name='0',

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-3,
            "architecture": "Multi Layer Perceptron",
            "dataset": "Mnist",
            "epochs": 5,
        }
    )
    model = MultilayerPerceptron(in_channel=28 * 28, num_layer=2, n_class=10).to(DEVICE)
    if LOAD_MODEL:
        load_checkpoint(torch.load(WEIGHT_DIR), model)

    optimizer = Lion(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, test_loader = get_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=1e-2,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=NUM_EPOCHS)
    metric_collection = metrics(DEVICE)
    wandb.watch(model, criterion=loss_fn, log="all", log_freq=2, log_graph=True)

    train_l, train_d, test_l, test_d = [], [], [], []
    patience = PATIENCE
    max_accuracy = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_fn(epoch, train_loader, model, optimizer, scheduler, loss_fn, scaler,
                                              metric_collection)
        # check accuracy
        test_loss, test_accuracy = eval_fn(test_loader, model, loss_fn, metric_collection, DEVICE)
        wandb.log({"train_loss": train_loss,
                   "train_accuracy": train_accuracy,
                   'test_loss': test_loss,
                   'test_accuracy': test_accuracy,
                   })
        train_l.append(train_loss)
        train_d.append(train_accuracy)
        test_l.append(test_loss)
        test_d.append(test_accuracy)
        # save model
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            patience = PATIENCE
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
        if patience == 0:
            break
        patience -= 1

    save_plot(train_l, train_d, test_l, test_d)
    wandb.finish()
    sys.exit()


if __name__ == "__main__":
    main()
