import os
import numpy as np
import tensorflow as tf
from snare.model import DeepScan
from snare.utils import load_dataset
import pickle

# === CONFIGURATION ===
MAXSEQ = 4980
NUM_FEATURE = 20
NUM_CLASSES = 2
BATCH_SIZE = 128
EPOCHS = 10

DATA_PATH = 'dataset/pssm.cv.csv'
MODEL_DIR = 'snare/models'
PIPELINE_PATH = os.path.join(MODEL_DIR, 'feature_pipeline.pkl')
WEIGHTS_PATH = os.path.join(MODEL_DIR, 'deepscan_model.weights.h5')

# === Create directories if needed ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load dataset ===
x_train, y_train = load_dataset(DATA_PATH)
x_train = np.reshape(x_train, [-1, 1, MAXSEQ, NUM_FEATURE])
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)

# === Build model ===
model = DeepScan()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train ===
model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# === Save model weights ===
model.save_weights(WEIGHTS_PATH)
print(f"[✔] Model weights saved to: {WEIGHTS_PATH}")

# === Save dummy pipeline placeholder ===
dummy_pipeline = {
    "note": "This is a placeholder. Replace with real preprocessing if needed."
}
with open(PIPELINE_PATH, "wb") as f:
    pickle.dump(dummy_pipeline, f)
print(f"[✔] Feature pipeline saved to: {PIPELINE_PATH}")