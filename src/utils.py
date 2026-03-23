"""Shared utilities: seeding, device selection, logging, and configuration."""

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

LABEL_HUMAN = 0
LABEL_AI = 1
LABEL_NAMES = {LABEL_HUMAN: "human", LABEL_AI: "ai"}


def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_is_usable() -> bool:
    """Return True only if CUDA is available AND functional.

    Runs a small tensor operation on the GPU to confirm the installed
    PyTorch build actually supports the hardware (avoids sm_XX mismatch).
    """
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.tensor([1.0], device="cuda")
        _ = t + t
        return True
    except Exception:
        return False


def get_device() -> torch.device:
    """Return the best available torch device.

    Falls back to CPU if CUDA is technically available but the GPU arch
    is not supported by the installed PyTorch build.
    """
    if cuda_is_usable():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project-level logger."""
    logger = logging.getLogger("hc3_detector")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@dataclass
class TrainConfig:
    """Configuration for transformer fine-tuning (Tier 2)."""

    model_name: str = "distilroberta-base"
    max_length: int = 512
    num_labels: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = False
    seed: int = 42
    early_stopping_patience: int = 2
    output_dir: str = str(MODELS_DIR / "roberta-hc3")
    logging_steps: int = 20


@dataclass
class BaselineConfig:
    """Configuration for the TF-IDF + Logistic Regression baseline (Tier 1)."""

    max_features: int = 50_000
    ngram_range: tuple = (1, 2)
    C: float = 1.0
    max_iter: int = 1000
    seed: int = 42
