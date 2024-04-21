import torch
from image_manipulation.models.vanilla_aec import LitVanillaAutoEncoder
from image_manipulation.models.conv_aec import LitConvAutoEncoder
from image_manipulation.models.vae import LitVAE


def load_model(name, lr):
    match name:
        case "vanilla_aec":
            model = LitVanillaAutoEncoder(lr, latent_dim=3)
        case "conv_aec":
            model = LitConvAutoEncoder(lr)
        case "vae":
            model = LitVAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=50, lr=lr)
    return model


def load_model_from_ckpt(name, weights):
    match name:
        case "vanilla_aec":
            model = LitVanillaAutoEncoder.load_from_checkpoint(
                weights, lr=0.001, latent_dim=3
            )
        case "conv_aec":
            model = LitConvAutoEncoder.load_from_checkpoint(weights, lr=0.001)
        case "vae":
            model = LitVAE.load_from_checkpoint(
                weights, x_dim=784, h_dim1=512, h_dim2=256, z_dim=50, lr=0.001
            )
    return model
