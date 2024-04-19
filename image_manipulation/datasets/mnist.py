from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


class MnistDataset:
    """A class for the MNIST dataloader"""

    def __init__(self, batch_size=256):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                transforms.Lambda(lambda x: x.to(self.device)),
            ]
        )
        self.train_ds = MNIST(
            "./", transform=self.img_transform, train=True, download=True
        )
        self.val_ds = MNIST(
            "./", transform=self.img_transform, train=False, download=True
        )

    def get_dataloaders(self):
        train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        return train_dl, val_dl
