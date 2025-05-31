# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 21:37:38 2021
@author: TMU
"""
import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd

maxseq = 4980

def generate_dataset(in_dir, out_file, label):
    expected_length = maxseq * 20  # 20 features per position

    for filename in os.listdir(in_dir):
        file_path = os.path.join(in_dir, filename)

        if os.path.isdir(file_path) or not filename.endswith('.pssm'):
            continue

        try:
            with open(file_path) as f:
                pssm = f.readlines()[3:-6]

            arr = []
            for line in pssm:
                tokens = line.strip().split()
                if len(tokens) >= 22:
                    arr.extend([float(k) for k in tokens[2:22]])

            # Pad if needed
            while len(arr) < expected_length:
                arr.extend([0.0] * 20)

            # Trim if somehow too long
            arr = arr[:expected_length]

            out_file.write(f"{label}," + ",".join(map(str, arr)) + "\n")

        except Exception as e:
            print(f"[Skipped] {filename}: {e}")
# === Dataset input folders ===
pos_dir = 'pssm/snare/cv'
neg_dir = 'pssm/non-snare/cv'

# === Output CSV ===
os.makedirs('dataset', exist_ok=True)
with open(os.path.join('dataset', 'pssm.cv.csv'), 'w') as ftrn:
    generate_dataset(pos_dir, ftrn, 1)
    generate_dataset(neg_dir, ftrn, 0)

print("✅ Dataset encoding completed.")