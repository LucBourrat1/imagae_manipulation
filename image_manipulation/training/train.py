import torch
import torch.nn as nn
from torch_snippets import *
from image_manipulation.datasets.mnist import MnistDataset
from image_manipulation.config.config import load_config
import argparse
from image_manipulation.scripts.utils import *


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="../config.yaml", help="path to the config.yaml file"
    )
    return parser.parse_args()


def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss


def train(cfg, trn_dl, val_dl):
    model = load_model(cfg.train.model)
    print(f"[INFO] Training {cfg.train.model} for {cfg.train.nb_epochs} epochs..")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    log = Report(cfg.train.nb_epochs)
    for epoch in range(cfg.train.nb_epochs):
        N = len(trn_dl)
        for ix, (data, _) in enumerate(trn_dl):
            loss = train_batch(data, model, criterion, optimizer)
            log.record(pos=(epoch + (ix + 1) / N), trn_loss=loss, end="\r")

        N = len(val_dl)
        for ix, (data, _) in enumerate(val_dl):
            loss = validate_batch(data, model, criterion)
            log.record(pos=(epoch + (ix + 1) / N), val_loss=loss, end="\r")
        log.report_avgs(epoch + 1)
    log.plot(log=True)
    torch.save(model.to("cpu").state_dict(), f"./{cfg.train.model}.pth")


if __name__ == "__main__":
    args = parser()
    cfg = load_config(args.cfg)
    trn_dl, val_dl = MnistDataset(batch_size=256).get_dataloaders()
    train(cfg, trn_dl, val_dl)
