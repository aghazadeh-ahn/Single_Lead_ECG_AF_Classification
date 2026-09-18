"""
scripts/evaluate.py
Evaluates a trained model checkpoint on the unseen test dataset.
Generates Challenge F1, Macro F1, Per-class metrics, and saves the Confusion Matrix.

Usage:
    python scripts/evaluate.py --model baseline --checkpoint experiments/checkpoints/best_baseline.pt
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import CHECKPOINT_DIR, DEVICE, RESULTS_DIR
from src.dataset import get_dataloaders
from src.metrics import evaluate_predictions, get_confusion_matrix
from src.utils import load_checkpoint, plot_confusion_matrix, set_seed
from scripts.train import get_model_instance


@torch.no_grad()
def evaluate_test_set(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for signals, labels in test_loader:
        signals = signals.to(device)
        logits = model(signals)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    metrics = evaluate_predictions(y_true, y_pred)
    cm = get_confusion_matrix(y_true, y_pred)
    return metrics, cm


def main(args):
    set_seed(42)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"🔍 Evaluating on device: {device}")

    # 1. Load Test DataLoader
    _, _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_filtering=args.use_filtering,
    )

    # 2. Instantiate and load model weights
    model = get_model_instance(args.model).to(device)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR / f"best_{args.model}.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)

    # 3. Perform Evaluation
    print("⏳ Running inference on Test Set...")
    metrics, cm = evaluate_test_set(model, test_loader, device)

    # 4. Print Clean Summary
    print("\n" + "=" * 50)
    print(f"📊 TEST EVALUATION RESULTS: {args.model.upper()}")
    print("=" * 50)
    print(f"⭐ Official Challenge F1 : {metrics['challenge_f1']:.4f}")
    print(f"🔹 Macro F1              : {metrics['macro_f1']:.4f}")
    print(f"🔹 Overall Accuracy      : {metrics['accuracy'] * 100:.2f}%")
    print("-" * 50)
    print(f"   - F1 Normal (N)       : {metrics['f1_normal_N']:.4f}")
    print(f"   - F1 AF (A)           : {metrics['f1_af_A']:.4f}")
    print(f"   - F1 Other (O)        : {metrics['f1_other_O']:.4f}")
    print(f"   - F1 Noisy (~)        : {metrics['f1_noisy_~']:.4f}")
    print("=" * 50)

    # 5. Save Artifacts (Metrics JSON & Confusion Matrix Image)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results_json_path = RESULTS_DIR / f"{args.model}_test_metrics.json"
    with open(results_json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    cm_image_name = f"{args.model}_confusion_matrix.png"
    plot_confusion_matrix(
        cm,
        save_name=cm_image_name,
        title=f"Confusion Matrix - {args.model.upper()} (Test Set)",
    )

    print(f"📁 Metrics saved to: {results_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set.")
    parser.add_argument("--model", type=str, default="baseline", help="Model name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint pt file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Num workers")
    parser.add_argument("--use_filtering", action="store_true", default=False, help="Apply software filters")

    args = parser.parse_args()
    main(args)
