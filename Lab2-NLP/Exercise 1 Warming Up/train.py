import sys
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from model import TinyLLama
from utils import (get_loaders, save_checkpoint, load_checkpoint,
                   load_best_model, metrics, eval_fn,
                   create_directory_if_does_not_exist, EarlyStopping, Lion)


def train_fn(epoch, loader, model, optimizer, scheduler, criterion, scaler, metric_collection, device):
    model.train()
    running_loss = 0

    for (data, target) in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # forward
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = criterion(prediction, target)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item() # .item porta il calcolo alla cpu, non riesco a farlo andare sulla GPU?

        # metrics
        prediction = (prediction > 0.5).float()
        metric_collection(prediction, target.int())

    train_loss = running_loss / len(loader)
    train_accuracy = metric_collection['BinaryAccuracy'].compute() * 100

    metric_collection.reset()

    return train_loss, train_accuracy


def main(wb, train_dir, test_dir, checkpoint_dir, weight_dir, device, num_workers):
    model = UNet(in_channels=wb.config['in_channels'], out_channels=wb.config['n_class']).to(device)

    optimizer = Lion(model.parameters(), lr=wb.config['learning_rate'], weight_decay=wb.config['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    metric_collection = metrics(wb, device)

    if wb.config['evaluation']:
        load_best_model(torch.load(weight_dir), model)
        test_loader = get_loaders(train_dir, test_dir, wb.config['batch_size'], num_workers, training=False)
        eval_fn(test_loader, model, criterion, metric_collection, device)
        sys.exit()

    elif wb.config['transfer learning']:
        print("=> Transfer Learning initialization")
        load_best_model(torch.load(''.join(["result/", wb.config['base model'], "/model.pth.tar"])), model)
        start = 0
        patience = EarlyStopping('max', wb.config['patience'])

    elif wb.resumed:
        start, monitored_value, count = load_checkpoint(torch.load(checkpoint_dir), model, optimizer)
        patience = EarlyStopping('max', wb.config['patience'], count, monitored_value)

    else:
        start = 0
        patience = EarlyStopping('max', wb.config['patience'])

    train_loader, test_loader = get_loaders(train_dir, test_dir, wb.config['batch_size'], num_workers)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=wb.config['max_lr'],
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=wb.config['num_epochs'] - start)

    wb.watch(model, log="all", log_graph=True)
    # nn.utils.clip_grad_norm_(model.parameters(), 5.0)

    for epoch in range(start, wb.config['num_epochs']):
        train_loss, train_accuracy, train_dice, train_iou = train_fn(epoch, train_loader, model, optimizer, scheduler,
                                                                     criterion, scaler,
                                                                     metric_collection, device)

        test_loss, test_accuracy, test_dice, test_iou = eval_fn(test_loader, model, criterion, metric_collection,
                                                                device)

        wb.log({"train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_dice": train_dice,
                "train_iou": train_iou,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'test_dice': test_dice,
                'test_iou': test_iou,
                'epoch': epoch + 1,
                })

        # save best model
        if patience(test_iou):
            wb.log({
                "dice_epoch": epoch + 1,
                "iou_epoch": epoch + 1,
            })
            checkpoint = {"state_dict": model.state_dict()}
            save_checkpoint("=> Best Model found ! Don't Stop Me Now", checkpoint, weight_dir)

        # save checkpoint
        checkpoint = {
            'start': epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "max_accuracy": getattr(patience, 'baseline'),
            "count": getattr(patience, 'count'),
        }
        save_checkpoint("=> Saving checkpoint", checkpoint, checkpoint_dir)
        # early stopping
        if getattr(patience, 'count') == 0:
            print("=> My Patience is Finished ! It's Time to Stop this Shit")
            break

    sys.exit()


if __name__ == "__main__":
    wab = wandb.init(
        # set the wandb project where this run will be logged
        project="Efficient Unet 2",
        # group='Experiment',
        tags=[],
        resume=False,
        name='version-7.2',
        config={
            # model parameters
            "architecture": "Unet",
            'in_channels': 3,
            'n_class': 1,

            # datasets
            "dataset": "COCO",

            # hyperparameters
            "learning_rate": 1e-4,
            "batch_size": 64,
            "optimizer": 'Lion',
            "weight_decay": 1e-2,
            "scheduler": "One Cycle Learning",
            "max_lr": 1e-4,
            "num_epochs": 1000,
            "patience": 20,

            # run type
            "evaluation": False,
            "transfer learning": False,
            "base model": '-'
        })
    # Local parameters
    check_dir = ''.join(["checkpoint/", wab.name, "/"])
    w_dir = ''.join(["result/", wab.name, "/"])
    train = "COCOdataset2017/annotations/instances_train2017.json"   # 'Dataset/train/images'
    test = "COCOdataset2017/annotations/instances_val2017.json"  # 'Dataset/val/images'
    create_directory_if_does_not_exist(check_dir, w_dir)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_workers = 4
    main(wab, train, test, check_dir, w_dir, dev, n_workers)
