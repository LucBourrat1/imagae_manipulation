from image_manipulation.models.vanilla_autoencoder import VanillaAutoEncoder
from torch_snippets import *
from image_manipulation.datasets.mnist import MnistDataset


def inference(model, val_ds):
    for i in range(3):
        ix = np.random.randint(len(val_ds))
        im, _ = val_ds[ix]
        _im = model(im[None])[0]
        fig, ax = plt.subplots(1, 2, figsize=(3, 3))
        show(im[0], ax=ax[0], title="input")
        show(_im[0], ax=ax[1], title="prediction")
        plt.tight_layout()
        plt.show()
        fig.savefig(f"./inf_{i}.png")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VanillaAutoEncoder(latent_dim=3)
    state_dict = torch.load("../training/vanilla_ae.pth")
    model.load_state_dict(state_dict)
    model.to(device)
    val_ds = MnistDataset(batch_size=256).val_ds

    ix = np.random.randint(len(val_ds))
    # im, _ = val_ds[ix]
    # print(im.shape)
    # _im = model(im[None])
    # print(_im.shape)
    inference(model, val_ds)
