import sys
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from model import MultilayerPerceptron
from utils import (get_loaders, save_checkpoint, load_checkpoint, load_best_model, metrics, eval_fn, Lion)

# Local parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 8
PIN_MEMORY = True
WEIGHT_DIR = "result/checkpoint.pth.tar"
CHECKPOINT_DIR = "checkpoint/checkpoint.pth.tar"


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


def main(wb):
    model = MultilayerPerceptron(in_channel=wb.config['in_channel'],
                                 num_layer=wb.config['num_layer'],
                                 n_class=wb.config['n_class']).to(DEVICE)

    optimizer = Lion(model.parameters(), lr=wb.config['learning_rate'], weight_decay=wb.config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    if wb.resumed:
        start, max_accuracy, patience = load_checkpoint(torch.load(CHECKPOINT_DIR), model, optimizer)
    else:
        start = 0
        max_accuracy = 0
        patience = wb.config['patience']

    metric_collection = metrics(DEVICE)

    if wb.config['evaluation']:
        load_best_model(torch.load(WEIGHT_DIR), model)
        test_loader = get_loaders(wb.config['batch_size'], NUM_WORKERS, PIN_MEMORY, training=False)
        eval_fn(test_loader, model, criterion, metric_collection, DEVICE)
        sys.exit()

    train_loader, test_loader = get_loaders(wb.config['batch_size'], NUM_WORKERS, PIN_MEMORY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=wb.config['max_lr'],
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=wb.config['num_epochs'] - start)

    wb.watch(model, criterion=criterion, log="all", log_freq=2, log_graph=True)

    for epoch in range(start, wb.config['num_epochs']):
        train_loss, train_accuracy = train_fn(epoch, train_loader, model, optimizer, scheduler, criterion, scaler,
                                              metric_collection)

        test_loss, test_accuracy = eval_fn(test_loader, model, criterion, metric_collection, DEVICE)

        wb.log({"train_loss": train_loss,
                "train_accuracy": train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                })

        # save best model
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            patience = wb.config['patience']  # reset patience
            checkpoint = {"state_dict": model.state_dict()}
            save_checkpoint("=> Best Model found ! Don't Stop Me Now", checkpoint, WEIGHT_DIR)
        # early stopping
        if patience == 0:
            print("=> My Patience is Finished ! It's Time to Stop this Shit")
            break
        else:
            patience -= 1

        # save checkpoint
        checkpoint = {
                'start': epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "max_accuracy": max_accuracy,
                "patience": patience,
            }
        save_checkpoint("=> Saving checkpoint", checkpoint, CHECKPOINT_DIR)

    sys.exit()


if __name__ == "__main__":
    main(wandb.init(
        # set the wandb project where this run will be logged
        project="A baseline MLP",
        group='MultiLayerPerceptron',
        tags=[],
        resume=False,
        name='experiment-1',

        config={
            # model parameters
            "architecture": "Multi Layer Perceptron",
            'in_channel': 28 * 28,
            'num_layer': 2,
            'n_class': 10,

            # datasets
            "dataset": "Mnist",

            # hyperparameters
            "learning_rate": 1e-3,
            "batch_size": 16384,
            "optimizer": 'Lion',
            "weight_decay": 1e-2,
            "scheduler": "One Cycle Learning",
            "max_lr": 1e-2,
            "num_epochs": 150,
            "patience": 20,

            # run type
            "evaluation": False,
        }
    ))
