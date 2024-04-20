from torch_snippets import *
from image_manipulation.datasets.mnist import MnistDataset
from image_manipulation.scripts.utils import load_model_from_ckpt
from image_manipulation.config.config import load_config
import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="../config.yaml", help="path to the config.yaml file"
    )
    return parser.parse_args()


def inference(model, val_ds):
    for i in range(3):
        ix = np.random.randint(len(val_ds))
        im, _ = val_ds[ix]
        _im = model(im[None])[0]
        fig, ax = plt.subplots(1, 2, figsize=(8, 8))
        show(im[0], ax=ax[0], title="input")
        show(_im[0], ax=ax[1], title="prediction")
        plt.tight_layout()
        plt.show()
        fig.savefig(f"./inf_{i}.png")


if __name__ == "__main__":
    args = parser()
    cfg = load_config(args.cfg)
    model = load_model_from_ckpt(cfg.inference.model, cfg.inference.weights)
    val_ds = MnistDataset(batch_size=256).val_ds
    inference(model, val_ds)
