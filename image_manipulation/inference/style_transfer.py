from torch_snippets import *
from torchvision import transforms as T
from torch.nn import functional as F
from torchvision.models import vgg19
import torch
import argparse
from PIL import Image
import yaml
import warnings

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="../config_st.yaml", help="path to the config file"
    )
    return parser.parse_args()


def load_config(file_path):
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_dict


preprocess = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x.mul_(255)),
    ]
)
postprocess = T.Compose(
    [
        T.Lambda(lambda x: x.mul_(1.0 / 255)),
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        ),
    ]
)


def get_loss_funcs(style_layers, content_layers):
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(
        content_layers
    )
    loss_fns = [loss_fn.to(DEVICE) for loss_fn in loss_fns]
    return loss_fns


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        feat = input.view(b, c, h * w)
        G = feat @ feat.transpose(1, 2)
        G.div_(h * w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = F.mse_loss(GramMatrix()(input), target)
        return out


class vgg19_modified(nn.Module):
    def __init__(self):
        super().__init__()
        features = list(vgg19(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x, layers=[]):
        order = np.argsort(layers)
        _results, results = [], []
        for ix, model in enumerate(self.features):
            x = model(x)
            if ix in layers:
                _results.append(x)
        for o in order:
            results.append(_results[o])
        return results if layers is not [] else x


def main(args):
    cfg = load_config(args.cfg)
    vgg = vgg19_modified().to(DEVICE)
    imgs = [
        Image.open(p).resize((512, 512)).convert("RGB")
        for p in [cfg["style"], cfg["content"]]
    ]
    style_image, content_image = [preprocess(img).to(DEVICE)[None] for img in imgs]
    opt_img = content_image.data.clone()
    opt_img.requires_grad = True
    style_layers = [0, 5, 10, 19, 28]
    content_layers = [21]
    loss_layers = style_layers + content_layers
    loss_fns = get_loss_funcs(style_layers, content_layers)
    style_weights = [1000 / n**2 for n in [64, 128, 256, 512, 512]]
    content_weights = [1]
    weights = style_weights + content_weights
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets
    max_iters = 500
    optimizer = optim.LBFGS([opt_img])
    log = Report(max_iters)
    global iters
    iters = 0
    print(f"[INFO] Performing style transfering for {max_iters} epochs...")
    while iters < max_iters:

        def closure():
            global iters
            iters += 1
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [
                weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)
            ]
            loss = sum(layer_losses)
            loss.backward()
            log.record(pos=iters, loss=loss, end="\r")
            return loss

        optimizer.step(closure)

    log.plot(log=True)
    with torch.no_grad():
        out_img = postprocess(opt_img[0]).permute(1, 2, 0)
        i = out_img.detach().to("cpu").numpy()
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    H, W = i.shape[:2]
    plt.subplots(figsize=(W * px, H * px))
    plt.imshow(i)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("output.png")


if __name__ == "__main__":
    args = parser()
    main(args)
