import argparse, os
import keras, mlflow
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from keras import layers

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--target", choices=["local", "dagshub"], default="local")
a = p.parse_args()

if a.target == "dagshub":
    load_dotenv()                                   # reads .env (Task 3.2)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
else:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # local database file
mlflow.set_experiment("22i0896_A2_Ops_toy")

keras.utils.set_random_seed(42)
(x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
x, xt = x / 255.0, xt / 255.0
model = keras.Sequential([keras.Input((28, 28)), layers.Flatten(),
                          layers.Dense(128, activation="relu"),
                          layers.Dense(10, activation="softmax")])
model.compile(optimizer=keras.optimizers.Adam(a.lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch + 1)


with mlflow.start_run(run_name=f"toy_lr_{a.lr}"):
    mlflow.log_params({"lr": a.lr, "epochs": a.epochs, "units": 128,
                       "batch_size": 128, "dataset": "fashion_mnist"})
    mlflow.set_tags({"purpose": "Task 3.1 toy lab", "target": a.target})
    h = model.fit(x, y, validation_split=0.2, epochs=a.epochs, batch_size=128,
                  callbacks=[EpochLogger()], verbose=2)
    loss, acc = model.evaluate(xt, yt, verbose=0)
    mlflow.log_metrics({"test_loss": loss, "test_accuracy": acc})

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h.history["loss"], label="train"); ax.plot(h.history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title(f"lr = {a.lr}"); ax.legend()
    mlflow.log_figure(fig, "loss_curve.png")          # the plot artifact
    mlflow.keras.log_model(model, name="model")       # the model artifact
    print(f"test accuracy {acc:.4f}")