import tensorflow as tf
import numpy as np

# Load model and dataset
model = tf.keras.models.load_model("models/lstm_pose.h5")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2f}")
