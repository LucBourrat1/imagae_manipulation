from torch_snippets import *
import lightning as L
from torchvision.utils import make_grid


def _ConvLayer(input_features, output_features):
    return nn.Sequential(
        nn.Conv2d(input_features, output_features, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(0.1, inplace=True),
    )


def _UpScale(input_features, output_features):
    return nn.Sequential(
        nn.ConvTranspose2d(
            input_features, output_features, kernel_size=2, stride=2, padding=0
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


class Reshape(nn.Module):
    def forward(self, input):
        output = input.view(-1, 1024, 4, 4)  # channel * 4 * 4
        return output


class ConvVae(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.encoder = nn.Sequential(
            _ConvLayer(3, 128),
            _ConvLayer(128, 256),
            _ConvLayer(256, 512),
            _ConvLayer(512, 1024),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 1024),
            nn.Linear(1024, 1024 * 4 * 4),
            Reshape(),
            _UpScale(1024, 512),
        )

        self.decoder_A = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.decoder_B = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.state = []
        self.epoch = -1

    def forward(self, x, select="A"):
        if select == "A":
            out = self.encoder(x)
            out = self.decoder_A(out)
        else:
            out = self.encoder(x)
            out = self.decoder_B(out)
        return out

    def configure_optimizers(self):
        optA = optim.Adam(
            [
                {"params": self.encoder.parameters()},
                {"params": self.decoder_A.parameters()},
            ],
            lr=5e-5,
            betas=(0.5, 0.999),
        )
        optB = optim.Adam(
            [
                {"params": self.encoder.parameters()},
                {"params": self.decoder_B.parameters()},
            ],
            lr=5e-5,
            betas=(0.5, 0.999),
        )
        return optA, optB

    def training_step(self, train_batch, batch_idx):
        optA, optB = self.optimizers()
        optA.zero_grad()
        optB.zero_grad()
        imgA, imgB, targetA, targetB = train_batch
        _imgA, _imgB = self.forward(imgA, "A"), self.forward(imgB, "B")
        lossA = nn.functional.l1_loss(_imgA, targetA)
        lossB = nn.functional.l1_loss(_imgB, targetB)

        lossA.backward()
        lossB.backward()

        optA.step()
        optB.step()
        self.log_dict({"lossA": lossA.item(), "lossB": lossB.item()})
        self.state.append(train_batch)
        return

    def on_train_epoch_end(self):
        self.epoch += 1
        bs = 5
        with torch.no_grad():
            last_batch = self.state[-1]
            a, b, A, B = last_batch
            _a = self.forward(a[:bs], "A")
            _b = self.forward(a[:bs], "B")
            sample = torch.cat([A[:bs], _a, _b])
            images = make_grid(sample.view(15, 3, 64, 64), nrow=5, padding=10)
            self.logger.experiment.add_image("A to B", images, self.global_step)

            _a = self.forward(b[:bs], "A")
            _b = self.forward(b[:bs], "B")
            sample = torch.cat([B[:bs], _a, _b])
            images = make_grid(sample.view(15, 3, 64, 64), nrow=5, padding=10)
            self.logger.experiment.add_image("B to A", images, self.global_step)
