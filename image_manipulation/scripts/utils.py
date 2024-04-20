import torch
from image_manipulation.models.vanilla_aec import LitVanillaAutoEncoder
from image_manipulation.models.conv_aec import LitConvAutoEncoder


def load_model(name, lr):
    match name:
        case "vanilla_aec":
            model = LitVanillaAutoEncoder(lr, latent_dim=3)
        case "conv_aec":
            model = LitConvAutoEncoder(lr)
    return model


def load_model_from_ckpt(name, weights):
    match name:
        case "vanilla_aec":
            model = LitVanillaAutoEncoder.load_from_checkpoint(
                weights, lr=0.0001, latent_dim=3
            )
        case "conv_aec":
            model = LitConvAutoEncoder.load_from_checkpoint(weights, lr=0.0001)
    return model
