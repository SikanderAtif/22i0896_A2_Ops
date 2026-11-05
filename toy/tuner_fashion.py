import keras, keras_tuner as kt, pandas as pd
from keras import layers

keras.utils.set_random_seed(42)
(x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
x, xt = x / 255.0, xt / 255.0

def build(hp):
    m = keras.Sequential([
        keras.Input((28, 28)), layers.Flatten(),
        layers.Dense(hp.Int("units", 32, 512, step=32), activation="relu"),
        layers.Dense(10, activation="softmax")])
    m.compile(optimizer=keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return m

tuner = kt.Hyperband(build, objective="val_accuracy", max_epochs=10, factor=3, seed=42,
                     directory="kt", project_name="fashion_toy", overwrite=True)
tuner.search(x, y, validation_split=0.2,
             callbacks=[keras.callbacks.EarlyStopping(patience=2)], verbose=1)

rows = [dict(trial=t.trial_id, units=t.hyperparameters.get("units"),
             lr=t.hyperparameters.get("lr"), epochs=t.hyperparameters.get("tuner/epochs"),
             val_accuracy=round(t.score, 4))
        for t in tuner.oracle.get_best_trials(num_trials=100)]
table = pd.DataFrame(rows); table.to_csv("toy/tuner_fashion_results.csv", index=False)
print(table.head(15).to_string(index=False))

best = tuner.get_best_hyperparameters(1)[0]
print("\nBEST:", best.values)
print("Test accuracy of best model:", tuner.get_best_models(1)[0].evaluate(xt, yt, verbose=0)[1])