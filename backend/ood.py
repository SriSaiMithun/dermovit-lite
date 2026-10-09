"""
Feature-distance out-of-distribution gate (layer 2).

The classifier's softmax is overconfident on images that are nothing like its
training data (measured: 96-99% "confidence" on a face photo, a colour wheel
and a screenshot, vs ~49% on a genuine lesion). So instead of trusting the
probabilities, we ask a different question: does this image's *feature
vector* (the 256-d CLS embedding from the model) look like the features of
real HAM10000 training images?

Method: class-conditional Gaussians with a shared covariance in embedding
space (the standard Mahalanobis OOD detector). The score of an image is its
smallest squared Mahalanobis distance to any class mean. Images whose score is
above a threshold - set at the 99th percentile of held-out HAM10000 scores, so
~1% of genuine images are expected to be rejected - are treated as
out-of-distribution.

The statistics come from artifacts/ood_stats.npz, produced by
fit_ood_detector.py (run once on Colab with the dataset). If that file is
absent the gate is simply disabled and the API says so in /health.
"""

import os

import numpy as np


class FeatureGate:
    def __init__(self, means: np.ndarray, precision: np.ndarray, threshold: float, meta: dict):
        self.means = means.astype(np.float64)          # (K, D)
        self.precision = precision.astype(np.float64)  # (D, D)
        self.threshold = float(threshold)
        self.meta = meta

    @classmethod
    def load_if_available(cls, path: str):
        if not os.path.exists(path):
            return None
        data = np.load(path, allow_pickle=False)
        meta = {k: (data[k].item() if data[k].shape == () else data[k].tolist())
                for k in data.files if k not in ("means", "precision", "threshold")}
        return cls(data["means"], data["precision"], float(data["threshold"]), meta)

    def score(self, embedding: np.ndarray) -> float:
        """Smallest squared Mahalanobis distance from `embedding` to any class mean."""
        x = np.asarray(embedding, dtype=np.float64).reshape(-1)
        diffs = self.means - x                                     # (K, D)
        d2 = np.einsum("kd,de,ke->k", diffs, self.precision, diffs)  # (K,)
        return float(d2.min())

    def is_in_distribution(self, embedding: np.ndarray):
        s = self.score(embedding)
        return s <= self.threshold, s
