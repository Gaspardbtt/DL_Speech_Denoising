import torch
import torch.nn as nn
import torch.nn.functional as F

#-------------------------------------------------------------------------------------

# CNN arch for Mask approche

class CNN_MASK(nn.Module):
    def __init__(self):
        super(CNN_MASK, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(8, 1, kernel_size=3, padding=1)  
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.sigmoid(self.conv8(x))
        return x
    

class DEEP_CNN(nn.Module):
    def __init__(self):
        super(DEEP_CNN, self).__init__()


        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(8, 1, kernel_size=3, padding=1)  
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.sigmoid(self.conv8(x))
        return x

#https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 8))
        self.down1 = (Down(8, 16))
        self.down2 = (Down(16, 32))
        self.down3 = (Down(32, 64))
        self.down4 = (Down(64, 128))

        factor = 2 if bilinear else 1
        self.up1 = (Up(128, 64 // factor, bilinear))
        self.up2 = (Up(64, 32 // factor, bilinear))
        self.up3 = (Up(32, 16 // factor, bilinear))
        self.up4 = (Up(16, 8 // factor, bilinear))

        self.outc = (OutConv(8, n_classes))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)

        x3 = self.down2(x2)
        x3 = self.dropout(x3)

        x4 = self.down3(x3)
        x4 = self.dropout(x4)

        x5 = self.down4(x4)
        x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        x = self.dropout(x)

        x = self.up2(x, x3)
        x = self.dropout(x)

        x = self.up3(x, x2)
        x = self.dropout(x)

        x = self.up4(x, x1)
        x = self.dropout(x)

        logits = self.sigmoid(self.outc(x))
        return logits

#-------------------------------------------------------------------------------------

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

""" Parts of the U-Net model """


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

# ----------------Temporal approach-------------------


class DemucsLike(nn.Module):
    def __init__(self, input_channels=1, hidden=64, lstm_hidden=128):
        super().__init__()

        # Encoder (32 → 64 → 128)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=8, stride=2, padding=3),  
            nn.ReLU(),
            nn.Conv1d(hidden, hidden*2, kernel_size=8, stride=2, padding=3),        
            nn.ReLU(),
            nn.Conv1d(hidden*2, hidden*4, kernel_size=8, stride=2, padding=3),   
            nn.ReLU()
        )

        # LSTM
        self.lstm = nn.LSTM(hidden*4, lstm_hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.linear_after_lstm = nn.Linear(lstm_hidden, hidden*4)

        # Decoder (128 → 64 → 32 → 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden*4, hidden*2, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden*2, hidden, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden, input_channels, kernel_size=8, stride=2, padding=3)
        )

    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # LSTM
        z_lstm = z.permute(0, 2, 1)
        z_lstm, _ = self.lstm(z_lstm)
        z_lstm = self.linear_after_lstm(z_lstm)
        z_lstm = z_lstm.permute(0, 2, 1)

        # Decode
        return self.decoder(z_lstm)
