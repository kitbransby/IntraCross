import torch
import torch.nn as nn
import torchvision
import os

class CalciumDetector(nn.Module):
    def __init__(self, num_classes, input_dim, encoder_weights):
        super(CalciumDetector, self).__init__()

        self.encoder = torchvision.models.resnet50(weights=None)
        self.encoder.conv1 = nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Extracting layers
        self.layer0 = nn.Sequential(self.encoder.conv1,
                                     self.encoder.bn1,
                                     self.encoder.relu)
        self.layer1 = nn.Sequential(self.encoder.maxpool,
                                     self.encoder.layer1)
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        # Decoder layers
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.final_upconv = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(65, 32, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

        self.self_attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)

        self.row_pred = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2), )

    def forward(self, x):
        x1 = self.layer0(x)  # Conv1 + BN + ReLU
        x2 = self.layer1(x1)  # Maxpool + Layer1
        x3 = self.layer2(x2)  # Layer2
        x4 = self.layer3(x3)  # Layer3
        x5 = self.layer4(x4)  # Layer4

        # Decoder
        d5 = self.relu(self.upconv1(x5))
        d5 = torch.cat((d5, x4), dim=1)
        d5 = self.relu(self.conv1(d5))

        d4 = self.relu(self.upconv2(d5))
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.relu(self.conv2(d4))

        d3 = self.relu(self.upconv3(d4))
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.relu(self.conv3(d3))

        d2 = self.relu(self.upconv4(d3))
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.relu(self.conv4(d2))

        d1 = self.relu(self.final_upconv(d2))
        d1 = torch.cat((d1, x), dim=1)  # Concatenate with the original input
        d1 = self.relu(self.final_conv(d1))

        # BS, 32, H, W -> BS, 32, H -> BS, H, 32 -> BS, H, 2
        pooled = d1.mean(dim=-1)
        pooled = torch.permute(pooled, (0, 2, 1))

        # Self-attention across rows
        attention_out, _ = self.self_attention(pooled, pooled, pooled)

        out = self.row_pred(attention_out)

        return out

