"""Pixel art and cartoons for CLIP: finding the art's grid, clean upscaling, padding, prompts."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clip_style as cs  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

PAL = [(20, 20, 30), (200, 60, 50), (240, 200, 80), (60, 140, 220)]


def sprite(w=16, h=12, seed=0):
    rnd = random.Random(seed)
    img = Image.new("RGB", (w, h), PAL[0])
    px = img.load()
    for y in range(2, h - 2):
        for x in range(3, w - 3):
            px[x, y] = PAL[1 + (x // 3 + y // 2 + rnd.randint(0, 1)) % 3]
    return img


def test_finds_the_grid_and_upscales_cleanly():
    art = sprite()
    shipped = art.resize((art.width * 3, art.height * 3), Image.NEAREST)     # how sprites are often shared
    assert cs.block_size(shipped) == 3 and cs.block_size(art) == 1
    assert cs.looks_like(shipped) == "pixel"
    out, used = cs.prepare(shipped, "auto", 224)
    assert used == "pixel" and out.size == (224, 224)
    # the default enlarges smoothly from the art's own grid (what CLIP reads best): new in-between colours
    assert len(out.getcolors(1 << 16)) > len(PAL)
    sharp, _ = cs.prepare(shipped, "pixel", 224, pixel_filter="nearest")
    assert sharp.size == (224, 224)
    # nearest: no blended colours, and padded with the background, not cropped: every art colour is still there
    assert {c for _, c in sharp.getcolors(1 << 16)} == {c for _, c in art.getcolors(1 << 16)}
    # bicubic from the grid is not the same as bicubic from the shipped blocks
    blocks = shipped.resize((224, 224), Image.BICUBIC)
    assert out.tobytes() != blocks.tobytes()


def test_photos_and_cartoons():
    rnd = random.Random(1)
    photo = Image.new("RGB", (320, 240))
    photo.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(320 * 240)])
    assert cs.looks_like(photo) == "photo"
    assert cs.prepare(photo, "auto")[1] == "photo" and cs.prepare(photo, "auto")[0].size == (320, 240)
    toon = Image.new("RGB", (400, 260), (250, 250, 245))
    d = ImageDraw.Draw(toon)
    d.ellipse((60, 40, 220, 200), fill=(240, 180, 40), outline=(0, 0, 0), width=4)
    d.rectangle((250, 80, 360, 220), fill=(80, 160, 230), outline=(0, 0, 0), width=4)
    toon = toon.filter(ImageFilter.GaussianBlur(0.8))           # antialiased edges: many colours, still flat
    assert cs.looks_like(toon) == "cartoon"
    out, used = cs.prepare(toon, "auto")
    assert used == "cartoon" and out.size == (400, 400) and out.getpixel((5, 5)) == toon.getpixel((0, 0))
    assert cs.prepare(photo, "off")[0] is not None


def test_templates_and_sizes():
    base = ["a photo of a {}."]
    assert cs.templates(base, "off") == base
    assert "pixel art of a {}." in cs.templates(base, "pixel") and len(cs.templates(base, "auto")) == 9

    class P:
        image_processor = type("IP", (), {"crop_size": {"height": 336, "width": 336}})()
    assert cs.target_size(P()) == 336
    try:
        cs.prepare(sprite(), "watercolour")
    except ValueError:
        return
    raise AssertionError
