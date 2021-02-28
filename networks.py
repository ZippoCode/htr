import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, first_half, back_half):
        up = self.up(first_half)
        merge = torch.cat((up, back_half), dim=1)
        return self.conv(merge)


class OutputConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):

    def __init__(self, in_channels=1):
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
        self.output = OutputConvolution(in_channels=64, out_channels=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
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

        output = self.output(dec1)
        return output


class DiscriminatorConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_normalization=True):
        super(DiscriminatorConvolution, self).__init__()
        if not batch_normalization:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        output = self.conv(x)
        return output


class Discriminator(nn.Module):

    def __init__(self, in_channels=2):
        super(Discriminator, self).__init__()
        self.conv1 = DiscriminatorConvolution(in_channels=in_channels, out_channels=64, stride=1,
                                              batch_normalization=False)
        self.conv2 = DiscriminatorConvolution(in_channels=64, out_channels=128)
        self.conv3 = DiscriminatorConvolution(in_channels=128, out_channels=256)
        self.conv4 = DiscriminatorConvolution(in_channels=256, out_channels=256)
        self.conv5 = DiscriminatorConvolution(in_channels=256, out_channels=256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, padding=1)

    def forward(self, input_one, input_two):
        concatenation = torch.cat([input_one, input_two], dim=1)
        conv1 = self.conv1(concatenation)
        conv1 = F.pad(conv1, (0, 1, 0, 1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        output = self.conv6(conv5)
        output = F.pad(output, (0, 1, 0, 1))
        return torch.sigmoid(output)
