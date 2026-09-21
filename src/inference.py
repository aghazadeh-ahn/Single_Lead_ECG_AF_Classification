"""
src/inference.py
Inference helpers including multi-crop / sliding-window TTA.
"""

import numpy as np
import torch
import torch.nn as nn

from src.config import TTA_STRIDE, TTA_WINDOW_LENGTH
from src.preprocessing import prepare_tta_windows


@torch.no_grad()
def predict_logits_tta(
    model: nn.Module,
    signal: np.ndarray,
    device: torch.device,
    apply_filter: bool = False,
    window_length: int = TTA_WINDOW_LENGTH,
    stride: int = TTA_STRIDE,
    max_batch_size: int = 32,
) -> torch.Tensor:
    """
    Average class logits over overlapping windows of one recording.

    Returns:
        1D tensor of shape [num_classes].
    """
    model.eval()
    windows = prepare_tta_windows(
        signal,
        apply_filter=apply_filter,
        window_length=window_length,
        stride=stride,
    )
    tensor = torch.from_numpy(windows).float().unsqueeze(1)  # [N, 1, L]

    logits_chunks = []
    for start in range(0, tensor.size(0), max_batch_size):
        batch = tensor[start : start + max_batch_size].to(device)
        logits_chunks.append(model(batch))

    logits = torch.cat(logits_chunks, dim=0)
    return logits.mean(dim=0)
