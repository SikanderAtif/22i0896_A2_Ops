from PIL import Image
from visionguard.imaging import normalize, MAX_SIDE
from visionguard.data import decode_and_resize
from visionguard.imaging import to_jpeg_bytes


def test_normalize_flattens_transparency_and_caps_size():
    im = Image.new("RGBA", (2000, 1000), (0, 0, 0, 0))      # fully transparent
    out = normalize(im)
    assert out.mode == "RGB"
    assert max(out.size) == MAX_SIDE                           # longest side capped at 1024
    assert out.getpixel((10, 10)) == (255, 255, 255)           # transparency became white


def test_decode_and_resize_shape():
    jpeg = to_jpeg_bytes(normalize(Image.new("RGB", (300, 150), (10, 120, 200))))
    x = decode_and_resize(jpeg, 224)
    assert tuple(x.shape) == (224, 224, 3)
    assert x.dtype.name == "uint8"