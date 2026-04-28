import os
import threading
from pathlib import Path

import torch
from transformers import BertTokenizer

from ml.drift_detector import DriftDetector
from ml.model import MODEL_CONFIGS, OptimalFNDModel

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MAX_LEN = int(os.getenv("MAX_LEN", "128"))

_RECENT_TEXTS: list = []
_RECENT_TEXTS_LOCK = threading.Lock()
_MAX_RECENT = 500


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, OptimalFNDModel] = {}
        self._lock = threading.Lock()
        self.tokenizer: BertTokenizer | None = None
        self.drift_detector: DriftDetector | None = None

    def load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def get_model(self, name: str) -> OptimalFNDModel:
        if name not in self._models:
            with self._lock:
                if name not in self._models:
                    self._load_model(name)
        return self._models[name]

    def _load_model(self, name: str):
        if name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {name}. Choose from {list(MODEL_CONFIGS)}"
            )
        cfg = MODEL_CONFIGS[name]
        model = OptimalFNDModel(**cfg)
        weight_path = MODEL_DIR / f"{name}.pt"
        if weight_path.exists():
            model.load_state_dict(
                torch.load(str(weight_path), map_location="cpu", weights_only=True)
            )
        model.eval()
        self._models[name] = model
        from api.metrics import ACTIVE_MODELS

        ACTIVE_MODELS.set(len(self._models))

    @property
    def loaded_models(self) -> list:
        return list(self._models.keys())

    def predict(self, text: str, model_name: str) -> tuple[str, float]:
        model = self.get_model(model_name)
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        label = "fake" if pred_idx == 0 else "real"

        with _RECENT_TEXTS_LOCK:
            _RECENT_TEXTS.append(text)
            if len(_RECENT_TEXTS) > _MAX_RECENT:
                _RECENT_TEXTS.pop(0)

        return label, confidence

    def initialize_drift_detector(self, reference_texts: list = None):
        """Initialize drift detector with reference corpus (ISOT-style formal news)."""
        if reference_texts is None:
            reference_texts = [
                "The president signed the infrastructure bill into law on Thursday",
                "Federal Reserve announces interest rate decision after policy meeting",
                "Scientists publish new research findings on climate change effects",
                "Stock markets close higher after positive economic data released",
                "Senate votes to pass bipartisan legislation on healthcare reform",
                "United Nations security council meets to discuss regional conflict",
                "Technology companies report quarterly earnings above analyst expectations",
                "Department of Justice announces new regulations for financial institutions",
                "World health organization releases updated guidelines on public health",
                "Congressional leaders reach agreement on government spending bill",
                "International summit addresses trade relations between major economies",
                "Supreme court issues ruling on landmark civil rights case today",
                "Central bank raises benchmark interest rate by quarter percentage point",
                "Government releases monthly employment statistics showing job growth",
                "Environmental agency proposes stricter emissions standards for industry",
            ] * 20  # 300 reference texts for stable KS baseline
        self.drift_detector = DriftDetector(reference_texts, threshold=0.1)

    def get_drift_score(self) -> float:
        if self.drift_detector is None:
            return 0.0
        with _RECENT_TEXTS_LOCK:
            texts = list(_RECENT_TEXTS)
        if len(texts) < 10:
            return self.drift_detector.last_score
        result = self.drift_detector.check_drift(texts)
        from api.metrics import DRIFT_SCORE

        DRIFT_SCORE.set(result["drift_score"])
        return result["drift_score"]


registry = ModelRegistry()
