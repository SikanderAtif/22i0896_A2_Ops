import platform, sys
import tensorflow as tf, numpy, pandas, PIL, sklearn, matplotlib, mlflow

print("OS        :", platform.platform())
print("Python    :", sys.version.split()[0])
for name, mod in [("TensorFlow", tf), ("Keras", tf.keras), ("NumPy", numpy),
                  ("pandas", pandas), ("Pillow", PIL), ("scikit-learn", sklearn),
                  ("matplotlib", matplotlib), ("MLflow", mlflow)]:
    print(f"{name:10}:", mod.__version__)
print("GPUs      :", tf.config.list_physical_devices("GPU"))
print("Built with CUDA:", tf.test.is_built_with_cuda())