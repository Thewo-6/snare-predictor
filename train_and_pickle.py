# snare-web/train_and_pickle.py

import os
import pickle
import numpy as np
from Bio import SeqIO

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import AACTransformer, DPCTransformer


# ─── Paths to your training FASTA files ────────────────────────────────────────
SNARE_PATH     = "data/train_snare.fasta"
NON_SNARE_PATH = "data/train_non_snare.fasta"


# ─── Load sequences & labels ────────────────────────────────────────────────────
seqs, labels = [], []
for path, label in [(SNARE_PATH, 1), (NON_SNARE_PATH, 0)]:
    for rec in SeqIO.parse(path, "fasta"):
        seqs.append(str(rec.seq).upper().strip())
        labels.append(label)

print(f"Training on {len(seqs)} sequences "
      f"({sum(labels)} SNARE, {len(labels)-sum(labels)} non-SNARE)")


# ─── Build a single Pipeline with parallel feature extractors ─────────────────
feature_union = FeatureUnion([
    ("aac", AACTransformer()),
    ("dpc", DPCTransformer()),
])

pipeline = Pipeline([
    ("features", feature_union),     # both AAC & DPC see raw seqs
    ("scale", StandardScaler()),     # scale all 420 features
    ("svm", SVC(kernel="linear", probability=True))
])

# ─── Fit & save ────────────────────────────────────────────────────────────────
pipeline.fit(seqs, labels)

os.makedirs("model_pipeline", exist_ok=True)
with open("model_pipeline/feature_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Pipeline trained & saved to model_pipeline/feature_pipeline.pkl")

