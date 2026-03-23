"""Evaluate trained models on the test set and produce metrics + visualizations.

Usage:
    # Evaluate Tier 1 baseline
    python -m src.evaluate --tier 1 --classifier lr

    # Evaluate Tier 2 RoBERTa
    python -m src.evaluate --tier 2 --model_path models/roberta-hc3-best
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model import load_baseline
from src.utils import (
    DATA_PROCESSED_DIR,
    LABEL_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    get_device,
    setup_logging,
)

logger = setup_logging()


# ---------------------------------------------------------------------------
# Shared metric computation
# ---------------------------------------------------------------------------


def compute_all_metrics(
    labels: np.ndarray, preds: np.ndarray, probs: np.ndarray | None = None
) -> dict:
    """Return a dict of all evaluation metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if probs is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            metrics["roc_auc"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(LABEL_NAMES.values()),
        yticklabels=list(LABEL_NAMES.values()),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Path,
    title: str = "ROC Curve",
) -> None:
    """Save an ROC curve plot."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved to %s", save_path)


def per_domain_breakdown(
    test_df: pd.DataFrame, preds: np.ndarray, probs: np.ndarray | None = None
) -> pd.DataFrame:
    """Compute metrics per domain and return as a DataFrame."""
    test_df = test_df.copy()
    test_df["pred"] = preds
    if probs is not None:
        test_df["prob"] = probs

    records = []
    for domain, group in test_df.groupby("source"):
        m = compute_all_metrics(
            group["label"].values,
            group["pred"].values,
            group["prob"].values if probs is not None else None,
        )
        m["domain"] = domain
        m["n_samples"] = len(group)
        records.append(m)

    breakdown = pd.DataFrame(records)
    cols = ["domain", "n_samples", "accuracy", "precision", "recall", "f1"]
    if "roc_auc" in breakdown.columns:
        cols.append("roc_auc")
    return breakdown[cols]


# ---------------------------------------------------------------------------
# Tier-specific evaluation
# ---------------------------------------------------------------------------


def evaluate_baseline(classifier: str = "lr") -> dict:
    """Evaluate the Tier 1 baseline on the test set."""
    pipeline = load_baseline(classifier)
    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")

    preds = pipeline.predict(test_df["text"].fillna(""))

    probs = None
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        probs = pipeline.predict_proba(test_df["text"].fillna(""))[:, 1]

    metrics = compute_all_metrics(test_df["label"].values, preds, probs)
    logger.info("Baseline (%s) test metrics: %s", classifier.upper(), metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"baseline_{classifier}"

    report = classification_report(
        test_df["label"], preds, target_names=["human", "ai"]
    )
    logger.info("Classification report:\n%s", report)

    plot_confusion_matrix(
        test_df["label"].values,
        preds,
        RESULTS_DIR / f"{tag}_confusion.png",
        title=f"Baseline ({classifier.upper()}) Confusion Matrix",
    )

    if probs is not None:
        plot_roc_curve(
            test_df["label"].values,
            probs,
            RESULTS_DIR / f"{tag}_roc.png",
            title=f"Baseline ({classifier.upper()}) ROC Curve",
        )

    breakdown = per_domain_breakdown(test_df, preds, probs)
    breakdown.to_csv(RESULTS_DIR / f"{tag}_per_domain.csv", index=False)
    logger.info("Per-domain breakdown:\n%s", breakdown.to_string(index=False))

    with open(RESULTS_DIR / f"{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def evaluate_roberta(model_path: str | None = None) -> dict:
    """Evaluate the Tier 2 RoBERTa model on the test set."""
    path = model_path or str(MODELS_DIR / "roberta-hc3-best")
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()

    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")

    all_preds = []
    all_probs = []
    batch_size = 32

    texts = test_df["text"].fillna("").tolist()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, truncation=True, max_length=512, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())

    preds = np.array(all_preds)
    probs = np.array(all_probs)
    labels = test_df["label"].values

    metrics = compute_all_metrics(labels, preds, probs)
    logger.info("RoBERTa test metrics: %s", metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "roberta"

    report = classification_report(labels, preds, target_names=["human", "ai"])
    logger.info("Classification report:\n%s", report)

    plot_confusion_matrix(
        labels, preds, RESULTS_DIR / f"{tag}_confusion.png", title="RoBERTa Confusion Matrix"
    )
    plot_roc_curve(
        labels, probs, RESULTS_DIR / f"{tag}_roc.png", title="RoBERTa ROC Curve"
    )

    breakdown = per_domain_breakdown(test_df, preds, probs)
    breakdown.to_csv(RESULTS_DIR / f"{tag}_per_domain.csv", index=False)
    logger.info("Per-domain breakdown:\n%s", breakdown.to_string(index=False))

    with open(RESULTS_DIR / f"{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HC3 text detector")
    parser.add_argument(
        "--tier", type=int, default=2, choices=[1, 2], help="1 = baseline, 2 = RoBERTa"
    )
    parser.add_argument("--classifier", type=str, default="lr", choices=["lr", "svm"])
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tier == 1:
        evaluate_baseline(args.classifier)
    else:
        evaluate_roberta(args.model_path)
