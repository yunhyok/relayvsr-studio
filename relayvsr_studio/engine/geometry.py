"""Spatial geometry of a native RelayVSR checkpoint (2x or 4x)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelayGeometry:
    scale: int

    def __post_init__(self):
        if self.scale not in (2, 4):
            raise ValueError("RelayVSR native scale must be 2 or 4")

    @property
    def latent_stride(self) -> int:
        """LR pixels per Wan latent cell (also per Flash token): 4 for 4x, 8 for 2x."""
        return 16 // self.scale

    @property
    def wan_stride(self) -> int:
        """LR pixels per Wan transformer token (patch 2 on the latent grid)."""
        return 2 * self.latent_stride

    @property
    def wan_unshuffle(self) -> int:
        """Pixel-unshuffle factor of the Wan LQ projection input."""
        return 16 if self.scale == 4 else 8

    @property
    def wan_upsample(self) -> int:
        """Bilinear upsample applied to the LR frame before the Wan LQ projection."""
        return 4 if self.scale == 4 else 1

    @property
    def flash_encoder_strides(self) -> tuple[int, int, int]:
        return (2, 2, self.latent_stride // 4)

    def padded_size(self, height: int, width: int) -> tuple[int, int]:
        stride = self.wan_stride
        return height + (-height) % stride, width + (-width) % stride

    def wan_tokens(self, height: int, width: int) -> tuple[int, int]:
        ph, pw = self.padded_size(height, width)
        return ph // self.wan_stride, pw // self.wan_stride

    def latent_grid(self, height: int, width: int) -> tuple[int, int]:
        ph, pw = self.padded_size(height, width)
        return ph // self.latent_stride, pw // self.latent_stride

    def output_size(self, height: int, width: int) -> tuple[int, int]:
        return height * self.scale, width * self.scale


__all__ = ["RelayGeometry"]
