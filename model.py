
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_features=3, out_channels=2, base_ch=64, img_size=512):
        super().__init__()
        self.img_size = img_size

        # Paramétrico a imagen inicial (expandir parámetros a un mapa)
        self.param_fc = nn.Sequential(
            nn.Linear(in_features, base_ch * img_size * img_size),
            nn.GELU()
        )

        # Encoder
        self.enc1 = UNetBlock(base_ch, base_ch)
        self.enc2 = UNetBlock(base_ch, base_ch*2)
        self.enc3 = UNetBlock(base_ch*2, base_ch*4)
        self.enc4 = UNetBlock(base_ch*4, base_ch*8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(base_ch*8, base_ch*16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = UNetBlock(base_ch*16, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = UNetBlock(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = UNetBlock(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = UNetBlock(base_ch*2, base_ch)

        # Salida
        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        # x: (batch, in_features)
        batch = x.shape[0]
        # Expande parámetros a un "mapa base"
        x = self.param_fc(x).view(batch, -1, self.img_size, self.img_size)  # (batch, base_ch, H, W)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder con skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out
    
import torch.nn as nn

import torch.nn as nn

class MixtoParam2Image(nn.Module):
    def __init__(self, in_features=3, out_channels=2, alto=512, ancho=512, hidden=1024):
        super().__init__()
        self.alto = alto
        self.ancho = ancho
        self.out_channels = out_channels
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels * (alto // 32) * (ancho // 32)),
            nn.GELU()
        )

        # Decoder con kernels grandes y stride 2 hasta llegar a 512x512
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 512, kernel_size=13, stride=2, padding=6, output_padding=1),  # 16 -> 32
            nn.BatchNorm2d(512),
            nn.GELU(),

            nn.ConvTranspose2d(512, 256, kernel_size=11, stride=2, padding=5, output_padding=1),  # 32 -> 64
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.ConvTranspose2d(256, 128, kernel_size=9, stride=2, padding=4, output_padding=1),  # 64 -> 128
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, 128, kernel_size=7, stride=2, padding=3, output_padding=1),  # 128 -> 256
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # 256 -> 512
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # mantener 512
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),  # salida final
            # No activación final → permite valores positivos/negativos
        )

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, self.out_channels, self.alto // 32, self.ancho // 32)
        x = self.decoder(x)
        return x

    
class MixtoParam2Image_antiguo(nn.Module):
    def __init__(self, in_features=3, out_channels=2, alto=512, ancho=512, hidden=1024):
        super().__init__()
        self.alto = alto
        self.ancho = ancho
        self.out_channels = out_channels
        
        # Encoder totalmente conectado
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels * (alto // 32) * (ancho // 32)),
            nn.GELU()
        )
        # Decoder más profundo con kernels más grandes
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 512, kernel_size=7, stride=2, padding=3, output_padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, kernel_size=7, stride=2, padding=3, output_padding=1),  # 32x32 -> 64x64
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),  # 64x64 -> 128x128
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1),  # 128x128 -> 256x256
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # 256x256 -> 512x512
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # 512x512 -> 512x512
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),  # 512x512 -> 512x512
        )

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, self.out_channels, self.alto // 32, self.ancho // 32)
        x = self.decoder(x)
        return x