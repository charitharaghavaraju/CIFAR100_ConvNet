import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetModel(nn.Module):
    def __init__(self):
        super(ConvNetModel, self).__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(2, 2)),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(2, 2)),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(2, 2)),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(3200, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.network(x)
        return x


if __name__ == '__main__':
    image_in = torch.randn(20, 3, 32, 32)
    model = ConvNetModel()
    out = model(image_in)
    print(out)


