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
    def __init__(self, input_channels=1, hidden=64, lstm_hidden=128, dropout_rate=0.2):
        super().__init__()
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=8, stride=2, padding=3),  
            nn.ReLU(),
            self.dropout, 
            
            nn.Conv1d(hidden, hidden*2, kernel_size=8, stride=2, padding=3),        
            nn.ReLU(),
            self.dropout,
            
            nn.Conv1d(hidden*2, hidden*4, kernel_size=8, stride=2, padding=3),   
            nn.ReLU(),
            self.dropout,
            
            nn.Conv1d(hidden*4, hidden*8, kernel_size=8, stride=2, padding=3),   
            nn.ReLU(),
        )

        # LSTM 
        self.lstm = nn.LSTM(hidden*8, lstm_hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.linear_after_lstm = nn.Linear(lstm_hidden, hidden*8)

        # Decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden*8, hidden*4, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
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
        
        # Dropout 
        z_lstm = self.dropout(z_lstm) 
        
        z_lstm = self.linear_after_lstm(z_lstm)
        z_lstm = z_lstm.permute(0, 2, 1)

        # Decode
        return self.decoder(z_lstm)
    

#---------------------Demucs simple-----------------------

class VerySimpleDemucs(nn.Module):
    def __init__(self, input_channels=1, hidden=64, lstm_hidden=128, dropout_rate=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)

        # -------- Encoder (sans dropout = plus rapide)
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=8, stride=2, padding=3),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(hidden, hidden*2, kernel_size=8, stride=2, padding=3),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(hidden*2, hidden*4, kernel_size=8, stride=2, padding=3),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(hidden*4, hidden*8, kernel_size=8, stride=2, padding=3),
            nn.ReLU()
        )

        # -------- LSTM (gros gain ici)
        self.lstm = nn.LSTM(hidden*8, lstm_hidden, num_layers=2, batch_first=True)
        self.linear_after_lstm = nn.Linear(lstm_hidden, hidden*8)

        # -------- Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(hidden*8, hidden*4, kernel_size=8, stride=2, padding=3),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(hidden*4, hidden*2, kernel_size=8, stride=2, padding=3),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(hidden*2, hidden, kernel_size=8, stride=2, padding=3),
            nn.ReLU()
        )
        self.dec1 = nn.ConvTranspose1d(hidden, input_channels, kernel_size=8, stride=2, padding=3)

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # LSTM
        z = e4.permute(0, 2, 1)
        z, _ = self.lstm(z)
        z = self.dropout(z)
        z = self.linear_after_lstm(z)
        z = z.permute(0, 2, 1)

        # Decode
        d4 = self.dec4(z) + e3
        d3 = self.dec3(d4) + e2
        d2 = self.dec2(d3) + e1
        out = self.dec1(d2)
        return out


#######More complex demucs architecture########

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Gated Convolution (GLU)
# --------------------------------------------------
class ConvGLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=8, stride=2, padding=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        a, b = self.conv(x).chunk(2, dim=1)
        return a * torch.sigmoid(b)


# --------------------------------------------------
# Encoder
# --------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvGLU(in_ch, out_ch),
            nn.GroupNorm(1, out_ch)
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------
# Decoder
# --------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()

        self.deconv = nn.ConvTranspose1d(
            in_ch, out_ch * 2,
            kernel_size=8,
            stride=2,
            padding=3
        )

        self.norm = nn.GroupNorm(1, out_ch)
        self.skip_proj = nn.Conv1d(skip_ch, out_ch, kernel_size=1)

    def forward(self, x, skip):
        a, b = self.deconv(x).chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        x = self.norm(x)

        skip = self.skip_proj(skip)
        x = x[..., :skip.size(-1)]

        return x + skip


# --------------------------------------------------
# LIGHT DEMUCS
# --------------------------------------------------
class SimpleDemucs(nn.Module):
    def __init__(
        self,
        input_channels=1,
        base_channels=32,
        depth=5,
        max_channels=256,
        lstm_hidden=256,
        dropout=0.1
    ):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        ch = input_channels
        encoder_channels = []

        # -------- Encoder
        for _ in range(depth):
            out_ch = min(base_channels, max_channels)
            self.encoders.append(EncoderBlock(ch, out_ch))
            encoder_channels.append(out_ch)
            ch = out_ch
            base_channels *= 2

        # -------- Bottleneck LSTM (raisonnable)
        self.lstm = nn.LSTM(
            input_size=ch,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.linear = nn.Linear(lstm_hidden * 2, ch)
        self.dropout = nn.Dropout(dropout)

        # -------- Decoder
        for skip_ch in reversed(encoder_channels):
            self.decoders.append(
                DecoderBlock(ch, skip_ch, skip_ch)
            )
            ch = skip_ch

        self.final = nn.Conv1d(ch, input_channels, kernel_size=1)

    def forward(self, x):
        orig_len = x.size(-1)

        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        z = x.permute(0, 2, 1)
        z, _ = self.lstm(z)
        z = self.dropout(z)
        z = self.linear(z)
        x = z.permute(0, 2, 1)

        for dec in self.decoders:
            x = dec(x, skips.pop())

        x = self.final(x)

        if x.size(-1) != orig_len:
            x = F.pad(x, (0, orig_len - x.size(-1)))

        return x


