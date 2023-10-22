import sys
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from model import TinyLLama
from utils import (get_loaders, save_checkpoint, load_checkpoint,
                   load_best_model, metrics, eval_fn,
                   create_directory_if_does_not_exist, EarlyStopping, Lion)


def train_fn(epoch, loader, model, optimizer, scheduler, criterion, scaler, device):
    model.train()
    running_loss = 0

    for (data, target) in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).view(-1)

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
        running_loss += loss.item()  # .item porta il calcolo alla cpu, non riesco a farlo andare sulla GPU?

    train_loss = running_loss / len(loader)

    return train_loss


def main(wb, train_dir, test_dir, checkpoint_dir, weight_dir, device, num_workers):
    model = TinyLLama(vocab_size=wb.config['vocab_size'],
                      embedding_size=wb.config['embedding_size'],
                      num_heads=wb.config['num_heads'],
                      num_layers=wb.config['num_layers'],
                      max_seq_len=wb.config['max_seq_len']).to(device)

    optimizer = Lion(model.parameters(), lr=wb.config['learning_rate'], weight_decay=wb.config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
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
        patience = EarlyStopping('min', wb.config['patience'], baseline=1000)

    train_loader, test_loader = get_loaders(train_dir, test_dir, wb.config['batch_size'], wb.config['max_seq_len'],
                                            num_workers)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=wb.config['max_lr'],
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=wb.config['num_epochs'] - start)

    wb.watch(model, log="all", log_graph=True)
    # nn.utils.clip_grad_norm_(model.parameters(), 5.0)

    for epoch in range(start, wb.config['num_epochs']):
        train_loss = train_fn(epoch, train_loader, model, optimizer, scheduler, criterion, scaler, device)

        test_loss = eval_fn(test_loader, model, criterion, device)

        wb.log({"train_loss": train_loss,
                'test_loss': test_loss,
                'epoch': epoch + 1,
                })

        # save best model
        if patience(test_loss):
            wb.log({
                "test_epoch": epoch + 1,
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
        project="Small Language Models",
        # group='Experiment',
        tags=[],
        resume=False,
        name='Experiments-1',
        config={
            # model parameters
            "architecture": "Transformer decoder only",
            'vocab_size': 5549,
            'embedding_size': 768,
            'num_heads': 16,
            'num_layers': 4,
            'max_seq_len': 128,

            # datasets
            "dataset": "Divina Commedia",

            # hyperparameters
            "learning_rate": 1e-4,
            "batch_size": 1024,
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
    train = 'Dataset/dataset.txt'
    test = 'Dataset/dataset.txt'
    create_directory_if_does_not_exist(check_dir, w_dir)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_workers = 1
    main(wab, train, test, check_dir, w_dir, dev, n_workers)
