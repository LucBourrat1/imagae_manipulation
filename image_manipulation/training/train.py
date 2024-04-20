from image_manipulation.config.config import load_config
from image_manipulation.datasets.mnist import MnistDataset
from image_manipulation.scripts.utils import load_model
import lightning as L
import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="../config.yaml", help="path to the config.yaml file"
    )
    return parser.parse_args()


def train(cfg, trn_dl, val_dl):
    model = load_model(cfg.train.model, cfg.train.learning_rate)
    trainer = L.Trainer(max_epochs=cfg.train.nb_epochs)
    trainer.fit(model=model, train_dataloaders=trn_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    args = parser()
    cfg = load_config(args.cfg)
    trn_dl, val_dl = MnistDataset(batch_size=256).get_dataloaders()
    train(cfg, trn_dl, val_dl)
