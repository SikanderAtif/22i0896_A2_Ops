"""Upload validation. The image lives only in memory and is never written to disk."""
import hashlib
import io
from flask import Request
from PIL import Image
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP", "GIF"}    # what the BYTES must really be
MAX_PIXELS = 40_000_000                                           # decompression-bomb guard


class MemoryOnlyRequest(Request):
    """Werkzeug normally spools uploads over 500 KB to a temporary file. Keep them in RAM instead."""
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


class UploadError(Exception):
    def __init__(self, status, title, message):
        super().__init__(message)
        self.status, self.title, self.message = status, title, message


def read_upload(file, max_bytes):
    if file is None or not file.filename:
        raise UploadError(400, "No file selected", "Please choose an image to upload.")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(415, "Unsupported file type",
                          f"'.{ext or '?'}' files are not accepted. Allowed: "
                          + ", ".join(sorted(ALLOWED_EXTENSIONS)) + ".")
    name = (secure_filename(file.filename) or f"upload.{ext}")[:120]   # strips paths, spaces, ../

    data = file.read(max_bytes + 1)
    if not data:
        raise UploadError(400, "Empty file", "The uploaded file contains no data.")
    if len(data) > max_bytes:
        raise UploadError(413, "File too large",
                          f"The limit is {max_bytes / 1024 / 1024:.0f} MB. Please upload a smaller image.")

    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise UploadError(413, "Image too large",
                                  f"{im.width}x{im.height} pixels exceeds the {MAX_PIXELS:,}-pixel limit.")
            im.verify()                                  # structure check
        with Image.open(io.BytesIO(data)) as im:
            im.load()                                    # full decode: catches truncated files
            fmt, w, h = im.format, im.width, im.height
    except UploadError:
        raise
    except Exception:
        raise UploadError(422, "Image could not be read",
                          "The file has an image extension, but its contents could not be decoded. "
                          "It may be corrupted, truncated, or not really an image.")
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(415, "Unsupported image format", f"{fmt} images are not supported.")

    return {"filename": name, "data": data, "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "width": w, "height": h, "format": fmt}