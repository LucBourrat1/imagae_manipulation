import lightning as L
from torch import nn
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F


class LitVAE(L.LightningModule):
    """An implementation of a variational autoencoder"""

    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, lr):
        super().__init__()
        self.d1 = nn.Linear(x_dim, h_dim1)
        self.d2 = nn.Linear(h_dim1, h_dim2)
        self.d31 = nn.Linear(h_dim2, z_dim)
        self.d32 = nn.Linear(h_dim2, z_dim)
        self.d4 = nn.Linear(z_dim, h_dim2)
        self.d5 = nn.Linear(h_dim2, h_dim1)
        self.d6 = nn.Linear(h_dim1, x_dim)
        self.lr = lr

    def encoder(self, x):
        h = F.relu(self.d1(x))
        h = F.relu(self.d2(h))
        return self.d31(h), self.d32(h)

    def sampling(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return eps.mul(std).add(mean)

    def decoder(self, z):
        h = F.relu(self.d4(z))
        h = F.relu(self.d5(h))
        return F.sigmoid(self.d6(h))

    def forward(self, x):
        mean, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mean, log_var)
        return self.decoder(z), mean, log_var

    def loss_function(self, recon_x, x, mean, log_var):
        RECON = F.mse_loss(recon_x, x.view(-1, 784), reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return RECON + KLD, RECON, KLD

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        recon, mean, log_var = self.forward(x)
        loss, mse, kld = self.loss_function(recon, x, mean, log_var)
        self.log_dict(
            {
                "train_loss": loss,
                "train_kld": kld,
                "train_recon": mse,
                "train_log_var": log_var.mean(),
                "train_mean": mean.mean(),
            }
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        recon, mean, log_var = self.forward(x)
        loss, mse, kld = self.loss_function(recon, x, mean, log_var)
        self.log_dict(
            {
                "val_loss": loss,
                "val_kld": kld,
                "val_recon": mse,
                "val_log_var": log_var.mean(),
                "val_mean": mean.mean(),
            }
        )
        return loss

    def on_validation_epoch_end(self):
        z = torch.randn(64, 50)
        sample = self.decoder(z)
        grid = make_grid(sample.view(64, 1, 28, 28))
        self.logger.experiment.add_image("output_images", grid, self.global_step)
