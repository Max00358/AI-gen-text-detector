"""Training entry point for both Tier 1 (baseline) and Tier 2 (RoBERTa).

Usage:
    # Tier 1 baseline
    python -m src.train --tier 1 --classifier lr

    # Tier 2 RoBERTa (HuggingFace Trainer)
    python -m src.train --tier 2

    # Tier 2 with custom hyperparameters
    python -m src.train --tier 2 --lr 3e-5 --epochs 3 --batch_size 32
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.dataset import build_hf_dataset
from src.model import build_roberta_model, train_baseline
from src.utils import (
    BaselineConfig,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    TrainConfig,
    cuda_is_usable,
    seed_everything,
    setup_logging,
)

logger = setup_logging()


class LossLoggingCallback(TrainerCallback):
    """Print training loss at each logging step so it isn't swallowed by tqdm."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is not None:
            lr = logs.get("learning_rate", 0)
            epoch = logs.get("epoch", 0)
            grad = logs.get("grad_norm", 0)
            logger.info(
                "step %d/%d | loss: %.4f | lr: %.2e | grad_norm: %.3f | epoch: %.2f",
                state.global_step,
                state.max_steps,
                loss,
                lr,
                grad,
                epoch,
            )


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1, and ROC-AUC for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }


def train_roberta(config: TrainConfig) -> None:
    """Fine-tune RoBERTa using HuggingFace Trainer with early stopping."""
    seed_everything(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = build_roberta_model(config)

    logger.info("Tokenizing datasets...")
    train_ds = build_hf_dataset(DATA_PROCESSED_DIR / "train.csv", tokenizer, config.max_length)
    val_ds = build_hf_dataset(DATA_PROCESSED_DIR / "val.csv", tokenizer, config.max_length)

    output_dir = config.output_dir
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    use_gpu = cuda_is_usable()
    steps_per_epoch = len(train_ds) // (config.batch_size * config.gradient_accumulation_steps)
    estimated_total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(config.warmup_ratio * estimated_total_steps)
    logger.info(
        "Device: %s | total_steps ~%d | warmup_steps: %d",
        "cuda" if use_gpu else "cpu",
        estimated_total_steps,
        warmup_steps,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=config.max_grad_norm,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=config.fp16 and use_gpu,
        use_cpu=not use_gpu,
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        save_total_limit=2,
        report_to="none",
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            ),
            LossLoggingCallback(),
        ],
    )

    logger.info("Starting RoBERTa fine-tuning...")
    trainer.train()

    best_path = str(MODELS_DIR / "roberta-hc3-best")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    logger.info("Best model saved to %s", best_path)

    metrics = trainer.evaluate()
    logger.info("Final validation metrics: %s", metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HC3 text detector")
    parser.add_argument(
        "--tier", type=int, default=2, choices=[1, 2], help="1 = baseline, 2 = RoBERTa"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="lr",
        choices=["lr", "svm"],
        help="Baseline classifier type (tier 1 only)",
    )
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.tier == 1:
        cfg = BaselineConfig(seed=args.seed)
        train_baseline(config=cfg, classifier=args.classifier)
    else:
        cfg = TrainConfig(
            model_name=args.model_name,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            seed=args.seed,
        )
        train_roberta(cfg)
