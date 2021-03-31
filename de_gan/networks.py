import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, first_half: torch.Tensor, second_half: torch.Tensor):
        up = self.up(first_half)
        merge = torch.cat([up, second_half], dim=1)
        out = self.conv(merge)
        return out


class OutputGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputGenerator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return out


class Generator(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(in_channels=in_channels, out_channels=64)
        self.encoder2 = Encoder(in_channels=64, out_channels=128)
        self.encoder3 = Encoder(in_channels=128, out_channels=256)
        self.encoder4 = Encoder(in_channels=256, out_channels=512)
        self.encoder5 = Encoder(in_channels=512, out_channels=1024)
        self.decoder4 = Decoder(in_channels=1024, out_channels=512)
        self.decoder3 = Decoder(in_channels=512, out_channels=256)
        self.decoder2 = Decoder(in_channels=256, out_channels=128)
        self.decoder1 = Decoder(in_channels=128, out_channels=64)
        self.final = OutputGenerator(in_channels=64, out_channels=out_channels)
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d()

    def forward(self, x: torch.Tensor):
        enc1 = self.encoder1(x)
        pool1 = self.max_pooling(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.max_pooling(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.max_pooling(enc3)

        enc4 = self.encoder4(pool3)
        drop4 = self.dropout(enc4)
        pool4 = self.max_pooling(drop4)

        enc5 = self.encoder5(pool4)
        drop5 = self.dropout(enc5)

        dec4 = self.decoder4(drop5, drop4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        out = self.final(dec1)
        return out


class ConvolutionDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=False):
        super(ConvolutionDiscriminator, self).__init__()
        if not batch_normalization:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, padding=1,
                          bias=False),
                nn.MaxPool2d(kernel_size=2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, padding=1,
                          bias=False),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return out


class OutputDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return out


class Discriminator(nn.Module):

    def __init__(self, in_channels=2, out_channels=1):
        super(Discriminator, self).__init__()
        self.conv1 = ConvolutionDiscriminator(in_channels=in_channels, out_channels=64)
        self.conv2 = ConvolutionDiscriminator(in_channels=64, out_channels=128, batch_normalization=True)
        self.conv3 = ConvolutionDiscriminator(in_channels=128, out_channels=256, batch_normalization=True)
        self.conv4 = ConvolutionDiscriminator(in_channels=256, out_channels=256, batch_normalization=True)
        self.final = OutputDiscriminator(in_channels=256, out_channels=out_channels)

    def forward(self, x_one: torch.Tensor, x_two: torch.Tensor):
        merge = torch.cat([x_one, x_two], dim=1)
        out = self.conv1(merge)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.final(out)
        return out
