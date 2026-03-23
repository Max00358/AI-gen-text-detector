"""End-to-end pipeline smoke test using synthetic data.

Verifies that every component (preprocessing, dataset, model, training, evaluation)
works together without needing to download the full HC3 dataset.

Usage:
    python -m tests.test_pipeline
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import flatten_hc3, split_by_question, save_splits
from src.dataset import HC3Dataset, build_hf_dataset
from src.model import build_baseline_pipeline, build_roberta_model
from src.train import compute_metrics
from src.evaluate import compute_all_metrics, per_domain_breakdown, plot_confusion_matrix
from src.utils import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    BaselineConfig,
    TrainConfig,
    seed_everything,
    get_device,
    setup_logging,
)

logger = setup_logging()


def create_synthetic_hc3(n_questions: int = 100) -> pd.DataFrame:
    """Generate a small synthetic HC3-like DataFrame for testing."""
    seed_everything(42)
    domains = ["finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"]
    rows = []
    for i in range(n_questions):
        domain = domains[i % len(domains)]
        question = f"Test question {i} about {domain}?"
        human_answers = [
            f"This is a human answer {j} for question {i}. "
            f"Humans write with varied style and occasional typos."
            for j in range(np.random.randint(1, 4))
        ]
        chatgpt_answers = [
            f"As an AI language model, I can provide information about question {i}. "
            f"Here is a comprehensive and well-structured response."
        ]
        rows.append({
            "question": question,
            "human_answers": human_answers,
            "chatgpt_answers": chatgpt_answers,
            "source": domain,
        })
    return pd.DataFrame(rows)


def test_preprocessing():
    """Test flatten + split on synthetic data."""
    logger.info("=" * 60)
    logger.info("TEST: Preprocessing pipeline")
    logger.info("=" * 60)

    raw_df = create_synthetic_hc3(100)
    logger.info("Synthetic raw data: %d rows", len(raw_df))

    flat_df = flatten_hc3(raw_df)
    assert len(flat_df) > 0, "Flattened dataframe is empty"
    assert set(flat_df.columns) >= {"text", "label", "question", "source"}
    assert flat_df["label"].isin([0, 1]).all(), "Labels must be 0 or 1"
    logger.info("  Flattened: %d samples, labels: %s", len(flat_df), flat_df["label"].value_counts().to_dict())

    train_df, val_df, test_df = split_by_question(flat_df, seed=42)
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
    all_questions = set(train_df["question"]) | set(val_df["question"]) | set(test_df["question"])
    train_questions = set(train_df["question"])
    val_questions = set(val_df["question"])
    test_questions = set(test_df["question"])
    assert train_questions.isdisjoint(val_questions), "Train/val question leakage"
    assert train_questions.isdisjoint(test_questions), "Train/test question leakage"
    assert val_questions.isdisjoint(test_questions), "Val/test question leakage"
    logger.info("  No question leakage across splits.")

    save_splits(train_df, val_df, test_df)
    assert (DATA_PROCESSED_DIR / "train.csv").exists()
    assert (DATA_PROCESSED_DIR / "val.csv").exists()
    assert (DATA_PROCESSED_DIR / "test.csv").exists()
    logger.info("  CSV files saved successfully.")
    logger.info("PASSED: Preprocessing pipeline\n")


def test_dataset():
    """Test PyTorch Dataset and HF Dataset from saved CSVs."""
    logger.info("=" * 60)
    logger.info("TEST: Dataset loading and tokenization")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    ds = HC3Dataset(DATA_PROCESSED_DIR / "train.csv", tokenizer, max_length=128)
    assert len(ds) > 0
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["labels"] in [0, 1]
    logger.info("  HC3Dataset: %d samples, first sample keys: %s", len(ds), list(sample.keys()))

    hf_ds = build_hf_dataset(DATA_PROCESSED_DIR / "train.csv", tokenizer, max_length=128)
    assert len(hf_ds) > 0
    assert "input_ids" in hf_ds.column_names
    assert "labels" in hf_ds.column_names
    logger.info("  HF Dataset: %d samples, columns: %s", len(hf_ds), hf_ds.column_names)
    logger.info("PASSED: Dataset loading and tokenization\n")


def test_baseline_model():
    """Test Tier 1 baseline pipeline (TF-IDF + LR)."""
    logger.info("=" * 60)
    logger.info("TEST: Tier 1 baseline (TF-IDF + LR)")
    logger.info("=" * 60)

    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")

    cfg = BaselineConfig(max_features=1000, ngram_range=(1, 1))
    pipeline = build_baseline_pipeline(cfg, classifier="lr")
    pipeline.fit(train_df["text"].fillna(""), train_df["label"])

    preds = pipeline.predict(val_df["text"].fillna(""))
    probs = pipeline.predict_proba(val_df["text"].fillna(""))[:, 1]
    metrics = compute_all_metrics(val_df["label"].values, preds, probs)
    logger.info("  Baseline LR metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    logger.info("PASSED: Tier 1 baseline\n")


def test_roberta_model():
    """Test Tier 2 RoBERTa model instantiation and single forward pass."""
    logger.info("=" * 60)
    logger.info("TEST: Tier 2 RoBERTa forward pass")
    logger.info("=" * 60)

    cfg = TrainConfig(model_name="roberta-base", num_labels=2)
    model = build_roberta_model(cfg)
    device = get_device()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    inputs = tokenizer(
        ["This is a human-written test sentence.", "As an AI model, I generate text."],
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    ).to(device)
    inputs["labels"] = torch.tensor([0, 1]).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.loss is not None, "Loss should not be None"
    assert outputs.logits.shape == (2, 2), f"Expected (2, 2) logits, got {outputs.logits.shape}"
    logger.info("  Loss: %.4f", outputs.loss.item())
    logger.info("  Logits shape: %s", outputs.logits.shape)
    logger.info("  Predictions: %s", outputs.logits.argmax(dim=-1).cpu().tolist())
    logger.info("PASSED: Tier 2 RoBERTa forward pass\n")


def test_compute_metrics():
    """Test the compute_metrics function used by the Trainer."""
    logger.info("=" * 60)
    logger.info("TEST: compute_metrics function")
    logger.info("=" * 60)

    logits = np.array([[2.0, -1.0], [-1.0, 2.0], [1.5, -0.5], [-0.5, 1.5]])
    labels = np.array([0, 1, 0, 1])

    metrics = compute_metrics((logits, labels))
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] == 1.0, "Perfect predictions should give 1.0 accuracy"
    logger.info("  Metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    logger.info("PASSED: compute_metrics\n")


def test_evaluation_helpers():
    """Test evaluation visualization and per-domain breakdown."""
    logger.info("=" * 60)
    logger.info("TEST: Evaluation helpers")
    logger.info("=" * 60)

    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    np.random.seed(42)
    fake_preds = np.random.randint(0, 2, size=len(test_df))
    fake_probs = np.random.rand(len(test_df))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        test_df["label"].values,
        fake_preds,
        RESULTS_DIR / "test_confusion.png",
        title="Smoke Test Confusion Matrix",
    )
    assert (RESULTS_DIR / "test_confusion.png").exists()

    breakdown = per_domain_breakdown(test_df, fake_preds, fake_probs)
    assert len(breakdown) > 0
    assert "domain" in breakdown.columns
    assert "f1" in breakdown.columns
    logger.info("  Per-domain breakdown:\n%s", breakdown.to_string(index=False))
    logger.info("PASSED: Evaluation helpers\n")


def main():
    logger.info("=" * 60)
    logger.info("PIPELINE SMOKE TEST -- START")
    logger.info("=" * 60)
    logger.info("")

    test_preprocessing()
    test_dataset()
    test_baseline_model()
    test_roberta_model()
    test_compute_metrics()
    test_evaluation_helpers()

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
