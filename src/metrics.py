"""
src/metrics.py
Evaluation metrics for ECG classification.
Implements the official PhysioNet/CinC Challenge 2017 F1 metric,
per-class precision/recall/F1, accuracy, and confusion matrix utilities.
"""

from typing import Dict, Union
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import CLASSES, IDX_TO_LABEL


def compute_challenge_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the official PhysioNet/CinC 2017 Challenge F1 score:
    F1_challenge = (F1_N + F1_A + F1_O) / 3

    Note: The Noisy ('~') class is excluded from the final averaged metric.
    """
    # Calculate per-class F1 scores (labels: 0=N, 1=A, 2=O, 3=~)
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    
    f1_n = f1_per_class[0]
    f1_a = f1_per_class[1]
    f1_o = f1_per_class[2]
    
    challenge_f1 = (f1_n + f1_a + f1_o) / 3.0
    return float(challenge_f1)


def evaluate_predictions(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of classification metrics.

    Args:
        y_true: Ground truth integer class indices.
        y_pred: Predicted integer class indices.

    Returns:
        dict containing challenge_f1, macro_f1, accuracy, and individual class F1 scores.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    
    # Official PhysioNet Challenge metric
    f1_challenge = (f1_per_class[0] + f1_per_class[1] + f1_per_class[2]) / 3.0
    
    # Overall Macro and Accuracy
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "challenge_f1": float(f1_challenge),
        "macro_f1": float(f1_macro),
        "accuracy": float(accuracy),
        "f1_normal_N": float(f1_per_class[0]),
        "f1_af_A": float(f1_per_class[1]),
        "f1_other_O": float(f1_per_class[2]),
        "f1_noisy_~": float(f1_per_class[3]),
    }
    return metrics


def get_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    """Returns the 4x4 confusion matrix ordered by [N, A, O, ~]."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
