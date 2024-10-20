import torch
import torch.nn as nn

class FlexibleColorizationNet(nn.Module):
    def __init__(self):
        super(FlexibleColorizationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, output_padding=(1, 1)),  # Adjust padding
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # Crop to desired size if necessary
        return x[:, :, :1200, :760]