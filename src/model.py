# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic Conv Blocks
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# -------------------------
# Generator
# -------------------------
class Generator(nn.Module):
    """
    Inputs:
        imgs: (B, K, 3, H, W)
        input_ids: (B, K, seq_len)
        attention_mask: (B, K, seq_len)
    Output:
        (B, 3, H, W)
    """
    def __init__(self, text_encoder, img_size=256, text_dim=768):
        super().__init__()
        self.text_encoder = text_encoder
        self.text_dim = text_dim

        # image encoder (shared for all K images)
        self.img_enc = nn.Sequential(
            ConvBlock(3, 64, norm=False),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )

        # fuse text embedding into spatial map
        self.text_proj = nn.Linear(text_dim, 512)

        # decoder
        self.dec = nn.Sequential(
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, imgs, input_ids, attention_mask):
        B, K, C, H, W = imgs.shape

        # ----- Encode images -----
        imgs = imgs.view(B * K, C, H, W)
        img_feats = self.img_enc(imgs)        # (B*K, 512, h, w)
        _, C2, h, w = img_feats.shape
        img_feats = img_feats.view(B, K, C2, h, w).mean(dim=1)

        # ----- Encode text -----
        input_ids = input_ids.view(B * K, -1)
        attention_mask = attention_mask.view(B * K, -1)

        with torch.no_grad():
            txt_out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0]   # CLS token

        txt_out = txt_out.view(B, K, -1).mean(dim=1)
        txt_feat = self.text_proj(txt_out).unsqueeze(-1).unsqueeze(-1)
        txt_feat = txt_feat.expand(-1, -1, h, w)

        # ----- Fuse -----
        fused = img_feats + txt_feat

        # ----- Decode -----
        out = self.dec(fused)
        return (out + 1) / 2   # map from [-1,1] → [0,1]


# -------------------------
# PatchGAN Discriminator
# -------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 64, norm=False),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
