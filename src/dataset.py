"""PyTorch Dataset wrappers and HuggingFace Dataset builders for HC3 splits."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.utils import DATA_PROCESSED_DIR, TrainConfig, setup_logging

logger = setup_logging()


class HC3Dataset(Dataset):
    """Token-level PyTorch Dataset that wraps a preprocessed CSV split."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        text_column: str = "text",
        label_column: str = "label",
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        logger.info("Loaded %d samples from %s", len(self.df), csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row[self.text_column],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": int(row[self.label_column]),
        }


def build_hf_dataset(csv_path: str | Path, tokenizer: AutoTokenizer, max_length: int = 512):
    """Build a HuggingFace Dataset from a CSV file, tokenized and ready for Trainer."""
    from datasets import Dataset as HFDataset

    df = pd.read_csv(csv_path)
    ds = HFDataset.from_pandas(df)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text", "question", "source"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    logger.info("Built HF dataset with %d samples from %s", len(ds), csv_path)
    return ds


def get_dataloaders(
    config: TrainConfig,
    tokenizer: AutoTokenizer | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from processed CSVs."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = HC3Dataset(DATA_PROCESSED_DIR / "train.csv", tokenizer, config.max_length)
    val_ds = HC3Dataset(DATA_PROCESSED_DIR / "val.csv", tokenizer, config.max_length)
    test_ds = HC3Dataset(DATA_PROCESSED_DIR / "test.csv", tokenizer, config.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "DataLoaders ready -- train: %d batches, val: %d batches, test: %d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader
