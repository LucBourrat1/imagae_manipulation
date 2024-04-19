import torch
import torch.nn as nn
from image_manipulation.models.vanilla_autoencoder import VanillaAutoEncoder
from torch_snippets import *
from image_manipulation.datasets.mnist import MnistDataset


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


def train(trn_dl, val_dl):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VanillaAutoEncoder(latent_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 10
    log = Report(num_epochs)
    for epoch in range(num_epochs):
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
    torch.save(model.to("cpu").state_dict(), "./vanilla_ae.pth")


if __name__ == "__main__":
    trn_dl, val_dl = MnistDataset(batch_size=256).get_dataloaders()
    train(trn_dl, val_dl)
