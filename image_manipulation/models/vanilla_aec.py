from torch import nn
import lightning as L
import torch
import torchvision


class LitVanillaAutoEncoder(L.LightningModule):
    """An implementation of a vanilla autoencoder"""

    def __init__(self, lr, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh(),
        )
        self.loss_func = nn.MSELoss()
        self.lr = lr

    def embeddings(self, x):
        x = x.view(len(x), -1)
        return self.encoder(x)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        output = x.view(len(x), 1, 28, 28)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        output = self.forward(x)
        loss = self.loss_func(output, x)
        self.log("train_loss", loss, on_epoch=True)
        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("input_images", grid, self.global_step)
            x = output[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("output_images", grid, self.global_step)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        output = self.forward(x)
        loss = self.loss_func(output, x)
        self.log("val_loss", loss, on_epoch=True)
        return loss
