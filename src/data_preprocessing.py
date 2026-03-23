"""Download, flatten, and split the HC3 dataset.

Usage:
    python -m src.data_preprocessing                    # default: download + process 'all'
    python -m src.data_preprocessing --subset finance   # process a single domain
    python -m src.data_preprocessing --skip_download    # reuse already-downloaded data
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from src.utils import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    LABEL_AI,
    LABEL_HUMAN,
    seed_everything,
    setup_logging,
)

logger = setup_logging()

VALID_SUBSETS = ["all", "reddit_eli5", "finance", "medicine", "open_qa", "wiki_csai"]

_HF_REPO = "Hello-SimpleAI/HC3"


def download_hc3(subset: str = "all") -> pd.DataFrame:
    """Download the HC3 JSONL file from HuggingFace and return it as a raw DataFrame.

    Uses hf_hub_download to fetch the raw JSONL file directly, avoiding the
    deprecated custom dataset script (HC3.py) that newer versions of the
    `datasets` library refuse to execute.
    """
    logger.info("Downloading HC3 subset '%s' from HuggingFace...", subset)
    filename = f"{subset}.jsonl"
    local_path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=filename,
        repo_type="dataset",
    )
    logger.info("Downloaded to cache: %s", local_path)

    records = []
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_RAW_DIR / f"{subset}.parquet"
    df.to_parquet(raw_path, index=False)
    logger.info("Raw data saved to %s (%d rows)", raw_path, len(df))
    return df


def flatten_hc3(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-answer rows into individual (text, label) samples."""
    rows: list[dict] = []
    for _, item in df.iterrows():
        question = item["question"]
        source = item.get("source", "unknown")
        for ans in item["human_answers"]:
            if isinstance(ans, str) and ans.strip():
                rows.append(
                    {
                        "text": ans.strip(),
                        "label": LABEL_HUMAN,
                        "question": question,
                        "source": source,
                    }
                )
        for ans in item["chatgpt_answers"]:
            if isinstance(ans, str) and ans.strip():
                rows.append(
                    {
                        "text": ans.strip(),
                        "label": LABEL_AI,
                        "question": question,
                        "source": source,
                    }
                )
    flat = pd.DataFrame(rows)
    logger.info(
        "Flattened to %d samples (human=%d, ai=%d)",
        len(flat),
        (flat["label"] == LABEL_HUMAN).sum(),
        (flat["label"] == LABEL_AI).sum(),
    )
    return flat


def split_by_question(
    df: pd.DataFrame,
    test_size: float = 0.30,
    val_ratio: float = 0.50,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Question-level stratified split into train / val / test.

    Ensures all answers derived from the same question stay in the same split
    (prevents data leakage).
    """
    questions = df[["question", "source"]].drop_duplicates(subset="question")

    train_q, temp_q = train_test_split(
        questions, test_size=test_size, stratify=questions["source"], random_state=seed
    )
    val_q, test_q = train_test_split(
        temp_q, test_size=val_ratio, stratify=temp_q["source"], random_state=seed
    )

    train_df = df[df["question"].isin(train_q["question"])].reset_index(drop=True)
    val_df = df[df["question"].isin(val_q["question"])].reset_index(drop=True)
    test_df = df[df["question"].isin(test_q["question"])].reset_index(drop=True)

    logger.info(
        "Split sizes -- train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """Persist train/val/test DataFrames as CSV files."""
    out = output_dir or DATA_PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = out / f"{name}.csv"
        split_df.to_csv(path, index=False)
        logger.info("Saved %s split to %s (%d rows)", name, path, len(split_df))


def run(subset: str = "all", skip_download: bool = False, seed: int = 42) -> None:
    """Full preprocessing pipeline: download -> flatten -> split -> save."""
    seed_everything(seed)

    if skip_download:
        raw_path = DATA_RAW_DIR / f"{subset}.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"{raw_path} not found. Run without --skip_download first."
            )
        logger.info("Loading cached raw data from %s", raw_path)
        raw_df = pd.read_parquet(raw_path)
    else:
        raw_df = download_hc3(subset)

    flat_df = flatten_hc3(raw_df)
    train_df, val_df, test_df = split_by_question(flat_df, seed=seed)
    save_splits(train_df, val_df, test_df)
    logger.info("Preprocessing complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess HC3 dataset")
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=VALID_SUBSETS,
        help="HC3 subset to download and process",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download and reuse cached raw data",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(subset=args.subset, skip_download=args.skip_download, seed=args.seed)
