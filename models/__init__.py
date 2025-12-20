"""Model architectures for news classification."""

from .simple_classifier import SimpleClassifier
from .cnn_classifier import CNNClassifier
from .transformer_model import RussianNewsClassifier, MultilingualBERTClassifier
from .advanced_transformers import (
    RoBERTaNewsClassifier,
    DistilBERTNewsClassifier,
    MultiHeadAttentionClassifier,
    EnsembleClassifier,
)
from .ensemble import (
    WeightedEnsemble,
    StackingEnsemble,
    VotingEnsemble,
    create_ensemble,
)
from .lightning_module import NewsClassificationModule
from .transformer_lightning import TransformerClassificationModule

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
    "NewsClassificationModule",
    "TransformerClassificationModule",
]

