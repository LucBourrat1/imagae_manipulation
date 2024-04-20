from image_manipulation.config.config import load_config
from image_manipulation.datasets.mnist import MnistDataset
from image_manipulation.scripts.utils import load_model
import torch
import lightning as L
import torch.nn as nn
import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="../config.yaml", help="path to the config.yaml file"
    )
    return parser.parse_args()


class LitConvAutoEncoder(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        embeddings = self.model.encoder(x)
        output = self.model.decoder(embeddings)
        return embeddings, output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        output = self.model(x)
        loss = self.loss_func(output, x)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        output = self.model(x)
        loss = self.loss_func(output, x)
        self.log("val_loss", loss, on_epoch=True)
        return loss


def train(cfg, trn_dl, val_dl):
    m = load_model(cfg.train.model)
    model = LitConvAutoEncoder(m)
    trainer = L.Trainer(max_epochs=cfg.train.nb_epochs)
    trainer.fit(model=model, train_dataloaders=trn_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    args = parser()
    cfg = load_config(args.cfg)
    trn_dl, val_dl = MnistDataset(batch_size=256).get_dataloaders()
    train(cfg, trn_dl, val_dl)
