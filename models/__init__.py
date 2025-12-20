"""Model architectures for news classification.

Important: keep this package lightweight at import time.
Render/Uvicorn must import `api.main:app` before binding the port; importing the
training stack (pytorch-lightning/torchmetrics/matplotlib) here can delay startup
and cause Render port-scan timeouts.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "SimpleClassifier",
    "CNNClassifier",
    "RussianNewsClassifier",
    "MultilingualBERTClassifier",
    "RoBERTaNewsClassifier",
    "DistilBERTNewsClassifier",
    "MultiHeadAttentionClassifier",
    "EnsembleClassifier",
    "WeightedEnsemble",
    "StackingEnsemble",
    "VotingEnsemble",
    "create_ensemble",
    # training-only (kept for convenience, but lazily imported)
    "NewsClassificationModule",
    "TransformerClassificationModule",
]

_LAZY: dict[str, tuple[str, str]] = {
    "SimpleClassifier": ("models.simple_classifier", "SimpleClassifier"),
    "CNNClassifier": ("models.cnn_classifier", "CNNClassifier"),
    "RussianNewsClassifier": ("models.transformer_model", "RussianNewsClassifier"),
    "MultilingualBERTClassifier": ("models.transformer_model", "MultilingualBERTClassifier"),
    "RoBERTaNewsClassifier": ("models.advanced_transformers", "RoBERTaNewsClassifier"),
    "DistilBERTNewsClassifier": ("models.advanced_transformers", "DistilBERTNewsClassifier"),
    "MultiHeadAttentionClassifier": ("models.advanced_transformers", "MultiHeadAttentionClassifier"),
    "EnsembleClassifier": ("models.advanced_transformers", "EnsembleClassifier"),
    "WeightedEnsemble": ("models.ensemble", "WeightedEnsemble"),
    "StackingEnsemble": ("models.ensemble", "StackingEnsemble"),
    "VotingEnsemble": ("models.ensemble", "VotingEnsemble"),
    "create_ensemble": ("models.ensemble", "create_ensemble"),
    # training-only
    "NewsClassificationModule": ("models.lightning_module", "NewsClassificationModule"),
    "TransformerClassificationModule": ("models.transformer_lightning", "TransformerClassificationModule"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module 'models' has no attribute {name!r}")
    module_name, attr_name = _LAZY[name]
    mod = import_module(module_name)
    return getattr(mod, attr_name)

