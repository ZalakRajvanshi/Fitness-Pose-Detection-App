# train_model.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load dataset
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Flatten (samples, features)
if len(X.shape) > 2:
    X = X.reshape(X.shape[0], X.shape[2])

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------- Logistic Regression Baseline ----------
print("\n=== Logistic Regression (5-Fold CV) ===")
log_reg = LogisticRegression(max_iter=1000)
scores = cross_val_score(log_reg, X, y, cv=5)
print("Cross-Validation Accuracy:", scores.mean())

# ---------- Train/Test Split for NN ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------- Neural Network ----------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=8,
    verbose=1
)

# Save model
model.save("models/dense_pose.h5")

# Plot accuracy vs epoch
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs Epoch")
plt.savefig("results/accuracy.png")
plt.close()

# Evaluate on test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()

report = classification_report(y_test, y_pred, digits=3)
with open("results/report.txt", "w") as f:
    f.write(report)

print("\n=== Neural Network Report ===")
print(report)
