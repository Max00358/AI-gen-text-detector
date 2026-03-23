"""Model definitions for Tier 1 (baseline) and Tier 2 (RoBERTa fine-tuning)."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification

from src.utils import (
    BaselineConfig,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    TrainConfig,
    setup_logging,
)

logger = setup_logging()


# ---------------------------------------------------------------------------
# Tier 1: TF-IDF + Logistic Regression / SVM Baseline
# ---------------------------------------------------------------------------


def build_baseline_pipeline(
    config: BaselineConfig | None = None,
    classifier: str = "lr",
) -> Pipeline:
    """Build a scikit-learn Pipeline for TF-IDF + classifier.

    Args:
        config: Baseline configuration. Uses defaults if None.
        classifier: "lr" for Logistic Regression, "svm" for LinearSVC.
    """
    cfg = config or BaselineConfig()
    tfidf = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        sublinear_tf=True,
    )
    if classifier == "svm":
        clf = LinearSVC(C=cfg.C, max_iter=cfg.max_iter, random_state=cfg.seed)
    else:
        clf = LogisticRegression(
            C=cfg.C, max_iter=cfg.max_iter, random_state=cfg.seed
        )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def train_baseline(
    config: BaselineConfig | None = None,
    classifier: str = "lr",
) -> Pipeline:
    """Train and save the Tier 1 baseline model."""
    cfg = config or BaselineConfig()
    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")

    pipeline = build_baseline_pipeline(cfg, classifier=classifier)
    logger.info("Training baseline (%s)...", classifier.upper())
    pipeline.fit(train_df["text"].fillna(""), train_df["label"])

    val_preds = pipeline.predict(val_df["text"].fillna(""))
    report = classification_report(val_df["label"], val_preds, target_names=["human", "ai"])
    logger.info("Baseline validation report:\n%s", report)

    save_path = MODELS_DIR / f"baseline_{classifier}.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Baseline model saved to %s", save_path)
    return pipeline


def load_baseline(classifier: str = "lr") -> Pipeline:
    """Load a saved baseline pipeline."""
    path = MODELS_DIR / f"baseline_{classifier}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Tier 2: Fine-tuned RoBERTa
# ---------------------------------------------------------------------------


def build_roberta_model(config: TrainConfig | None = None):
    """Instantiate a RoBERTa model for sequence classification."""
    cfg = config or TrainConfig()
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Loaded %s -- %s total params, %s trainable",
        cfg.model_name,
        f"{n_params:,}",
        f"{n_trainable:,}",
    )
    return model
