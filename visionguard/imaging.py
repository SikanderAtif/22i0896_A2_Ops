"""Single source of truth for image normalization: used in cleaning AND serving."""
import io
from PIL import Image, ImageOps

MAX_SIDE = 1024
JPEG_QUALITY = 95

def normalize(im: Image.Image) -> Image.Image:
    """First frame, EXIF rotation, alpha flattened on white, RGB, longest side <= 1024."""
    if getattr(im, "is_animated", False):
        im.seek(0)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)  # only shrinks; keeps aspect
    return im

def to_jpeg_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()