from datetime import datetime

import numpy as np
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer


class DriftDetector:
    def __init__(
        self, reference_texts: list, n_features: int = 50, threshold: float = 0.1
    ):
        self.threshold = threshold
        self.n_features = n_features
        self.vectorizer = TfidfVectorizer(max_features=n_features, stop_words="english")
        ref_matrix = self.vectorizer.fit_transform(reference_texts)
        self.reference_arr = ref_matrix.toarray()
        self.last_score: float = 0.0
        self.drift_detected: bool = False
        self.last_check: str = datetime.utcnow().isoformat()

    def check_drift(self, new_texts: list) -> dict:
        new_matrix = self.vectorizer.transform(new_texts).toarray()
        ks_stats = []
        for i in range(self.reference_arr.shape[1]):
            stat, _ = ks_2samp(self.reference_arr[:, i], new_matrix[:, i])
            ks_stats.append(stat)
        mean_ks = float(np.mean(ks_stats))
        self.last_score = mean_ks
        self.drift_detected = mean_ks > self.threshold
        self.last_check = datetime.utcnow().isoformat()
        return {
            "drift_score": mean_ks,
            "drift_detected": self.drift_detected,
            "threshold": self.threshold,
            "features_checked": len(ks_stats),
            "checked_at": self.last_check,
        }

    def update_reference(self, new_texts: list):
        new_matrix = self.vectorizer.transform(new_texts).toarray()
        self.reference_arr = new_matrix
        self.drift_detected = False
        self.last_score = 0.0
