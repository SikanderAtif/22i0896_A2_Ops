"""Loads the champion ONCE and turns uploaded image bytes into a verdict."""
import io
import json
import sys
import time
from pathlib import Path
import tensorflow as tf
import keras
from PIL import Image
from visionguard.imaging import normalize, to_jpeg_bytes
from visionguard.data import decode_and_resize


class Predictor:
    def __init__(self, model_path, metadata_path=None, threshold=None):
        if not Path(model_path).exists():
            raise RuntimeError(f"Model file not found at '{model_path}'. "
                               "Run `python -m scripts.export_champion` or set MODEL_PATH.")
        t0 = time.perf_counter()
        self.model = keras.models.load_model(model_path)
        meta = {}
        if metadata_path and Path(metadata_path).exists():
            meta = json.loads(Path(metadata_path).read_text())
        self.model_name = meta.get("model_name", Path(model_path).stem)
        self.model_version = str(meta.get("version", "unknown"))
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        self.img_size = int(self.model.input_shape[1])
        # warm-up: the first call builds the TensorFlow graph, so do it now, not on a user's request
        self.model(tf.zeros((1, self.img_size, self.img_size, 3)), training=False)
        self.load_seconds = time.perf_counter() - t0

    def preprocess(self, data: bytes):
        """Exactly the training path: normalize -> JPEG q95 -> decode -> resize (antialias)."""
        with Image.open(io.BytesIO(data)) as im:
            jpeg = to_jpeg_bytes(normalize(im))
        return tf.cast(decode_and_resize(jpeg, self.img_size), tf.float32)[None]

    def predict(self, data: bytes) -> dict:
        t0 = time.perf_counter()
        p = float(self.model(self.preprocess(data), training=False)[0, 0])
        t = self.threshold
        is_ai = p >= t
        # confidence = how far p is from the threshold, scaled to 0..1 on that side
        confidence = (p - t) / (1 - t) if is_ai else (t - p) / t
        return {"label": "AI-generated" if is_ai else "Real", "p_ai": p,
                "confidence": confidence, "threshold": t,
                "model_name": self.model_name, "model_version": self.model_version,
                "latency_ms": (time.perf_counter() - t0) * 1000}

    def info(self) -> dict:
        return {"name": self.model_name, "version": self.model_version,
                "threshold": round(self.threshold, 4), "img_size": self.img_size}


if __name__ == "__main__":
    pr = Predictor("model/champion.keras", "model/metadata.json")
    print(f"Model {pr.model_name} v{pr.model_version} loaded ONCE in {pr.load_seconds:.2f} s "
          f"(threshold {pr.threshold:.4f}, input {pr.img_size}x{pr.img_size})\n")
    for path in sys.argv[1:]:
        r = pr.predict(Path(path).read_bytes())
        print(f"{path}\n  -> {r['label']:13s} p(AI)={r['p_ai']:.4f}  "
              f"confidence={r['confidence']:.1%}  ({r['latency_ms']:.0f} ms)")