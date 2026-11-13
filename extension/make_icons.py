#!/usr/bin/env python3
"""Draw the toolbar icons: a slim blade, grey when nothing is known about the page's model, glowing blue when
there is data (lit). Run after changing the design; the PNGs are committed."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent / "icons"


def blade(size: int, lit: bool) -> Image.Image:
    S = 512                                                  # draw large, then scale down
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shape = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(shape)
    d.polygon([(256, 22), (296, 120), (286, 330), (226, 330), (216, 120)], fill=255)      # leaf-shaped blade
    d.rounded_rectangle((140, 330, 372, 362), 14, fill=255)                                 # crossguard
    d.rounded_rectangle((238, 362, 274, 450), 10, fill=255)                                 # grip
    d.ellipse((226, 444, 286, 504), fill=255)                                               # pommel
    if lit:
        glow = shape.filter(ImageFilter.GaussianBlur(28))
        img.paste((30, 120, 255, 255), (0, 0), glow.point(lambda v: min(255, v * 2)))
        img.paste((120, 185, 255, 255), (0, 0), shape)
        edge = Image.new("L", (S, S), 0)
        ImageDraw.Draw(edge).line([(256, 40), (256, 320)], fill=255, width=10)              # bright edge
        img.paste((225, 240, 255, 255), (0, 0), edge)
    else:
        img.paste((120, 126, 136, 255), (0, 0), shape)
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for lit in (False, True):
        for s in (16, 32, 48, 128):
            blade(s, lit).save(OUT / f"{'lit' if lit else 'idle'}-{s}.png", optimize=True)
