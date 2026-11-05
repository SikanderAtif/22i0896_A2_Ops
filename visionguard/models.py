import keras
from keras import layers

METRICS = lambda: ["accuracy",
                   keras.metrics.Precision(name="precision"),
                   keras.metrics.Recall(name="recall"),
                   keras.metrics.AUC(name="auc")]


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=METRICS())


def build_baseline_cnn(img_size=224, dropout=0.3, filters=(32, 64, 128, 256)):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="scale_0_1")(inputs)
    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


# ---------- transfer learning ----------
LAST_BLOCK_PREFIX = {"mobilenetv2": "block_16", "efficientnetb0": "block7a"}


def build_transfer(backbone="mobilenetv2", img_size=224, dropout=0.3):
    shape = (img_size, img_size, 3)
    inputs = keras.Input(shape=shape, name="image")
    if backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
        # identical to mobilenet_v2.preprocess_input: [0,255] -> [-1,1]
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)
    elif backbone == "efficientnetb0":
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=shape)
        x = inputs   # EfficientNet has its own scaling layers built in and expects [0,255]
    else:
        raise ValueError(backbone)
    base.trainable = False
    x = base(x, training=False)          # keep BatchNorm in inference mode, even when fine-tuning
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_transfer"), base


def last_block_size(base, backbone):
    prefix = LAST_BLOCK_PREFIX[backbone]
    first = next(i for i, l in enumerate(base.layers) if l.name.startswith(prefix))
    return len(base.layers) - first


def unfreeze_last(base, n, freeze_bn=True):
    """Make only the last n layers of the backbone trainable (BatchNorm stays frozen)."""
    base.trainable = True
    cut = len(base.layers) - n
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cut and not (freeze_bn and isinstance(layer, layers.BatchNormalization))