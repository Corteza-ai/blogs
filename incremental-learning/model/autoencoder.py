"""This file contains the main classes to build an autoencoder."""

__all__ = ["Encoder", "Decoder", "AutoEncoder"]

import torch
from torch import nn


class Encoder(nn.Module):
    """Creates a neural network containing 5 convolutional layers
    and its subsequent layers BN and ReLU.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 128) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Extract latent features from input images.

        Args:
            imgs (torch.Tensor): A batch of images with size (B, C_in, H, W).
        Returns:
            torch.Tensor: The latent representation with size (B, C_out, H_out, W_out).
        """
        latent_z = self.layer1(imgs)
        latent_z = self.layer2(latent_z)
        latent_z = self.layer3(latent_z)
        latent_z = self.layer4(latent_z)
        latent_z = self.layer5(latent_z)
        return latent_z


class Decoder(nn.Module):
    """A model that applies a series of deconvolution operations."""

    def __init__(self, in_channels: int = 128, out_channels: int = 3) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 64, 3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, 24, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(24, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(
                16, out_channels, 3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Applies a series of deconvolution operators.
        Args:
            latent (torch.Tensor): The latent representations of a batch of images.
        Returns:
            torch.Tensor: The reconstructed images.
        """
        recons = self.layer1(latent)
        recons = self.layer2(recons)
        recons = self.layer3(recons)
        recons = self.layer4(recons)
        recons = self.layer5(recons)
        return recons


class AutoEncoder(nn.Module):
    """An AE model that applies a combination of an encoder and a decoder function."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Combines an encoder and a decoder function.
        Args:
            imgs (torch.Tensor): A batch of images with size (B, C, H, W)
        Returns:
            torch.Tensor: The reconstructed images.
        """
        latent_z = self.encoder(imgs)
        x_hat = self.decoder(latent_z)
        return x_hat
