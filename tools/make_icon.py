"""Generate the application icon (PNG + multi-size ICO) with Pillow."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw


def make(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = size * 0.06
    radius = size * 0.22
    # Background: rounded square with a vertical gradient (indigo -> cyan).
    gradient = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(gradient)
    top, bottom = (44, 62, 160), (22, 170, 200)
    for y in range(size):
        t = y / (size - 1)
        color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3)) + (255,)
        gdraw.line([(0, y), (size, y)], fill=color)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([margin, margin, size - margin, size - margin], radius=radius, fill=255)
    img.paste(gradient, (0, 0), mask)

    # Two frames: a small "LR" frame bottom-left relayed to a large "HR" frame top-right.
    w = size * 0.11
    small = [size * 0.20, size * 0.56, size * 0.44, size * 0.74]
    large = [size * 0.40, size * 0.22, size * 0.82, size * 0.56]
    draw.rounded_rectangle(large, radius=size * 0.03, outline=(255, 255, 255, 255), width=int(w * 0.5))
    draw.rounded_rectangle(small, radius=size * 0.02, outline=(255, 255, 255, 220), width=int(w * 0.4))
    # Arrow from small to large.
    ax0, ay0 = size * 0.36, size * 0.60
    ax1, ay1 = size * 0.56, size * 0.42
    draw.line([(ax0, ay0), (ax1, ay1)], fill=(255, 220, 90, 255), width=int(w * 0.45))
    head = size * 0.085
    draw.polygon([(ax1 + head * 0.15, ay1 - head * 0.15), (ax1 - head * 0.75, ay1 - head * 0.05), (ax1 + head * 0.05, ay1 + head * 0.75)], fill=(255, 220, 90, 255))
    # Keyframe dots along the bottom (sparse relay).
    for i, x in enumerate((0.22, 0.34, 0.46, 0.58, 0.70, 0.82)):
        r = size * (0.028 if i % 5 == 0 else 0.016)
        color = (255, 220, 90, 255) if i % 5 == 0 else (255, 255, 255, 200)
        cx, cy = size * x, size * 0.86
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return img


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "relayvsr_studio" / "app" / "resources"
    out.mkdir(parents=True, exist_ok=True)
    icon = make(512)
    icon.save(out / "icon.png")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    icon.save(out / "icon.ico", sizes=[(s, s) for s in sizes])
    print("wrote", out / "icon.png", out / "icon.ico")


if __name__ == "__main__":
    main()
