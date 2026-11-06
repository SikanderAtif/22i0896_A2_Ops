"""tf.data input pipeline. decode_and_resize() is ALSO used by the Flask app at serving time."""
import tensorflow as tf
import keras
from keras import layers

IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE


def decode_and_resize(image_bytes, img_size=IMG_SIZE):
    """JPEG bytes -> uint8 tensor (img_size, img_size, 3). Aspect ratio is NOT kept (squash)."""
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def build_augmenter(seed=42):
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.05, fill_mode="reflect", seed=seed),   # about ±18°
        layers.RandomZoom(0.10, fill_mode="reflect", seed=seed),
        layers.RandomContrast(0.10, seed=seed),
    ], name="augmentation")


# visionguard/data.py: delete "import pandas as pd" from the top, and instead:
def make_dataset(csv_path, img_size=IMG_SIZE, batch_size=32, training=False,
                 augment=True, seed=42):
    import pandas as pd          # only needed for training, keeps the serving image smaller
    df = pd.read_csv(csv_path)
    paths = df["filepath"].to_numpy()
    labels = df["y"].to_numpy(dtype="float32").reshape(-1, 1)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))                    # 1. list
    ds = ds.map(lambda p, y: (decode_and_resize(tf.io.read_file(p), img_size), y),
                num_parallel_calls=AUTOTUNE)                                     # 2. read+decode+resize
    ds = ds.cache()                                                              # 3. cache (uint8, small)
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)       # 4. shuffle
    ds = ds.batch(batch_size)                                                    # 5. batch
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if training and augment:                                                     # 6. augment (train only)
        aug = build_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)                                                 # 7. prefetch