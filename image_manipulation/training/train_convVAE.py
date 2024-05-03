from image_manipulation.config.config import load_config
from image_manipulation.datasets.faces import FacesDataset
from image_manipulation.models.conv_vae import ConvVae
from pytorch_lightning.loggers import TensorBoardLogger
import lightning as lit
from torch_snippets import *
import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="../config.yaml", help="path to the config.yaml file"
    )
    return parser.parse_args()


def train(trn_dl):
    logger = TensorBoardLogger("tb_logs", name="convVAE")
    model = ConvVae()
    trainer = lit.Trainer(logger=logger, max_epochs=1000, log_every_n_steps=4)
    trainer.fit(model=model, train_dataloaders=trn_dl)


if __name__ == "__main__":
    a = FacesDataset(
        Glob("../datasets/cropped_faces_personA"),
        Glob("../datasets/cropped_faces_personB"),
    )
    x = DataLoader(a, batch_size=32, collate_fn=a.collate_fn)
    train(x)
