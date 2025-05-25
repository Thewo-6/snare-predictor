# ─── snare-web/features.py ───────────────────────────────────────────────────────────

import numpy as np
from sklearn.base import TransformerMixin

class AACTransformer(TransformerMixin):
    """
    Amino Acid Composition (AAC).
    Transforms a list of sequences (strings) into a 2D array of shape
    (n_samples, 20), where each column is the relative frequency
    of one of the 20 standard amino acids.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Coerce to plain Python strings
        seqs = [str(s) for s in X]

        aa_order = list("ACDEFGHIKLMNPQRSTVWY")
        out = []
        for seq in seqs:
            s = seq.strip().upper()
            L = len(s) if len(s) > 0 else 1
            freqs = [s.count(aa) / L for aa in aa_order]
            out.append(freqs)

        return np.array(out)


class DPCTransformer(TransformerMixin):
    """
    Dipeptide Composition (DPC).
    For each sequence, computes the frequency of each possible dipeptide
    (XX, XA, X[C|D|…], A[X|A|…], …, Y[Y]) in the same 20×20 alphabet,
    giving 400 features per sample.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Coerce to plain Python strings
        seqs = [str(s) for s in X]

        aa_order = list("ACDEFGHIKLMNPQRSTVWY")
        pairs = [a + b for a in aa_order for b in aa_order]
        out = []

        for seq in seqs:
            s = seq.strip().upper()
            L = len(s) - 1 if len(s) > 1 else 1
            # count overlapping dipeptides
            counts = {pp: 0 for pp in pairs}
            for i in range(len(s) - 1):
                di = s[i:i+2]
                if di in counts:
                    counts[di] += 1
            freqs = [counts[pp] / L for pp in pairs]
            out.append(freqs)

        return np.array(out)

