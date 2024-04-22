from torch_snippets import inspect, show, np, torch, nn
from torchvision.models import resnet50
from torchvision import transforms as T
from tqdm import trange
from torch.nn import functional as F
from PIL import Image
import argparse
import requests
import json
import os

model = resnet50(weights="ResNet50_Weights.DEFAULT")
for param in model.parameters():
    param.requires_grad = False
model = model.eval()

image_net_classes = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
image_net_classes = requests.get(image_net_classes).text
image_net_ids = eval(image_net_classes)
image_net_classes = {i: j for j, i in image_net_ids.items()}

imagenet_file = "imagenet_classes.json"
if not os.path.exists(imagenet_file):
    with open(imagenet_file, "w") as f:
        json.dump(image_net_classes, f, indent=2)

normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
denormalize = T.Normalize(
    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def list_of_strings(arg):
    return arg.split(",")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path", type=str, default="./elephant.jpg", help="path to the image"
    )
    parser.add_argument(
        "--labels",
        type=list_of_strings,
        default="lemon",
        help="list of labels separated by a ','",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default="No",
        help="'Yes' for prediction from image, 'No' for attack",
    )
    return parser.parse_args()


def image2tensor(input):
    x = normalize(input.clone().permute(2, 0, 1) / 255.0)[None]
    return x


def tensor2image(input):
    x = (denormalize(input[0].clone()).permute(1, 2, 0) * 255.0).type(torch.uint8)
    return x


def predict_on_image(input):
    model.eval()
    # show(input)
    input = image2tensor(input)
    pred = model(input)
    pred = F.softmax(pred, dim=-1)[0]
    prob, clss = torch.max(pred, 0)
    clss = image_net_ids[clss.item()]
    print(f"PREDICTION: `{clss}` @ {prob.item()}")


def attack(image, model, target, epsilon=1e-6):
    input = image2tensor(image)
    input.requires_grad = True
    pred = model(input)
    loss = nn.CrossEntropyLoss()(pred, target)
    loss.backward()
    # losses.append(loss.mean().item())
    output = input - epsilon * input.grad.sign()
    output = tensor2image(output)
    del input
    return output.detach()


def main(args):
    original_image = Image.open(args.img_path)
    original_image = np.array(original_image)
    original_image = torch.Tensor(original_image)

    if args.predict == "No":
        modified_images = []
        # desired_targets = ["lemon", "comic book", "sax, saxophone"]
        desired_targets = args.labels

        for target in desired_targets:
            str_target = target
            target = torch.tensor([image_net_classes[target]])
            image_to_attack = original_image.clone()
            for _ in trange(15):
                image_to_attack = attack(image_to_attack, model, target)
            modified_images.append(image_to_attack)
            img = Image.fromarray(np.array(image_to_attack))
            img_name = args.img_path.split("/")[-1].replace(".jpg", "")
            img_name += f"_{str_target.replace(' ','')}.png"
            img.save(img_name)
        for idx, image in enumerate([original_image, *modified_images]):
            predict_on_image(image)
            img_name = args.img_path.split("/")[-1].replace(".jpg", "")
            img = Image.fromarray(np.array(modified_images[0]))
        # for target, image in zip(desired_targets, modified_images):
        #     img.save(f"{img_name}_{target.strip()}.png")
    else:
        predict_on_image(original_image)


if __name__ == "__main__":
    args = parser()
    main(args)
