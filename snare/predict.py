# snare/predict.py

import numpy as np
import tensorflow as tf
from snare.model import DeepScan
from snare.utils import generate_pssm_from_fasta

# Constants
MAXSEQ = 4980
NUM_FEATURE = 20
NUM_CLASSES = 2

# === Build and Load Model ===
model = DeepScan(
    num_filters=256,
    num_hidden=128,
    window_sizes=[8, 16, 24, 32, 40, 48]  # match training
)
_ = model(tf.random.normal([1, 1, MAXSEQ, NUM_FEATURE])) # Build the model by calling it once with dummy input
model.load_weights("snare/logs/weights.09.weights.h5")  # Adjust if you use another weights file

# === Prediction Interface ===
def predict_snare(fasta_seq):
    """
    Predict whether a sequence is SNARE or NON-SNARE.
    
    Args:
        fasta_seq (str): A FASTA-formatted string.
    
    Returns:
        dict: { "class": int, "probability": List[float] } or { "error": str }
    """
    try:
        features = generate_pssm_from_fasta(fasta_seq)  # -> np.array shape (MAXSEQ*NUM_FEATURE,)
        if features.shape != (MAXSEQ * NUM_FEATURE,):
            raise ValueError(f"Expected feature shape ({MAXSEQ * NUM_FEATURE},), got {features.shape}")

        features = np.reshape(features, [1, 1, MAXSEQ, NUM_FEATURE])
        probs = model.predict(features, verbose=0)[0]
        label = int(np.argmax(probs))
        return {"class": label, "probability": probs.tolist()}

    except Exception as e:
        return {"error": str(e)}
