import torch
from image_manipulation.models.vanilla_aec import VanillaAutoEncoder
from image_manipulation.models.conv_aec import ConvAutoEncoder


def load_model(name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    match name:
        case "vanilla_aec":
            model = VanillaAutoEncoder(latent_dim=3).to(device)
        case "conv_aec":
            model = ConvAutoEncoder().to(device)
    return model
