import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = [64, 128, 256, 512]) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return torch.sigmoid(self.final_conv(x))


def build_unet_model(device: torch.device | None = None) -> UNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    return model.to(device)


def load_unet_model(model_path: Path, device: torch.device | None = None) -> UNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet_model(device)
    model.has_trained_weights = False
    if model_path.exists():
        try:
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.has_trained_weights = True
        except Exception:
            pass
    return model


def segment_image(model: UNet, image: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    image_tensor = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.35).astype(np.uint8) * 255
    return mask


@dataclass
class DefectQuantification:
    area: float
    spread: float
    surface_occupancy: float


def calculate_quantification(mask: np.ndarray) -> DefectQuantification:
    pixels = mask > 0
    area = float(np.sum(pixels))
    spread = float(np.count_nonzero(pixels) / mask.size) if mask.size else 0.0
    surface_occupancy = float(spread * 100.0)
    return DefectQuantification(area=area, spread=spread, surface_occupancy=surface_occupancy)


def save_unet_weights(model: UNet, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
