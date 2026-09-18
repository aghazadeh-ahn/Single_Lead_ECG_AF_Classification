"""
src/utils.py
Common helper utilities:
- Reproducibility (seed setting)
- Model checkpoint saving and loading
- Logging & plotting evaluation confusion matrices
"""

import os
import random
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.config import CHECKPOINT_DIR, CLASS_NAMES, FIGURES_DIR, SEED


def set_seed(seed: int = SEED) -> None:
    """
    Sets random seeds across Python, NumPy, and PyTorch for full reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Random seed fixed to: {seed}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_challenge_f1: float,
    model_name: str = "best_model.pt",
    save_dir: Union[str, Path] = CHECKPOINT_DIR,
) -> Path:
    """
    Saves model weights and training metadata.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / model_name

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_challenge_f1": val_challenge_f1,
    }
    torch.save(state, save_path)
    print(f" Model saved: {save_path} (Val F1: {val_challenge_f1:.4f})")
    return save_path


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    """
    Loads model weights from a saved checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f" Loaded checkpoint from {checkpoint_path} (Epoch {checkpoint.get('epoch', 'N/A')})")
    return checkpoint


def plot_confusion_matrix(
    cm: np.ndarray,
    save_name: str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    """
    Plots and saves a styled Confusion Matrix heatmap.
    """
    labels = [f"{k}\n({CLASS_NAMES[k]})" for k in ["N", "A", "O", "~"]]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" Confusion matrix figure saved at: {save_path}")
