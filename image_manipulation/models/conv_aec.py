from torch import nn
import torch
import lightning as L


class LitConvAutoEncoder(L.LightningModule):
    """An implementation of a convolutional autoencoder"""

    def __init__(self, lr):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1),
            nn.Tanh(),
        )
        self.loss_func = nn.MSELoss()
        self.lr = lr

    def embeddings(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        output = self.forward(x)
        loss = self.loss_func(output, x)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        output = self.forward(x)
        loss = self.loss_func(output, x)
        self.log("val_loss", loss, on_epoch=True)
        return loss
