import sys
import mlflow, tensorflow as tf
from mlflow import MlflowClient
from PIL import Image
from visionguard import tracking
from visionguard.imaging import normalize, to_jpeg_bytes
from visionguard.data import decode_and_resize

path = sys.argv[1]
tracking.setup("dagshub")
c = MlflowClient()
mv = c.get_model_version_by_alias(tracking.REGISTERED_MODEL, "champion")
threshold = c.get_run(mv.run_id).data.metrics.get("decision_threshold", 0.5)
model = mlflow.keras.load_model(f"models:/{tracking.REGISTERED_MODEL}@champion")
size = model.input_shape[1]                      # 224 or 160, read from the model itself

with Image.open(path) as im:                     # identical preprocessing to training
    jpeg = to_jpeg_bytes(normalize(im))
x = tf.cast(decode_and_resize(jpeg, size), tf.float32)[None]
p = float(model.predict(x, verbose=0)[0, 0])
label = "AI-generated" if p >= threshold else "Real"
print(f"Champion: {tracking.REGISTERED_MODEL} version {mv.version} (run {mv.run_id[:8]})")
print(f"Image: {path}\np(AI) = {p:.4f}  threshold = {threshold:.4f}  ->  {label} "
      f"(confidence {max(p, 1 - p):.1%})")