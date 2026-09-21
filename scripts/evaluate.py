"""
scripts/evaluate.py
Evaluates a trained model checkpoint on the unseen test dataset.
Supports single-crop inference and sliding-window TTA (logit averaging).

Usage:
    python scripts/evaluate.py --model crnn --checkpoint experiments/checkpoints/best_crnn.pt
    python scripts/evaluate.py --model crnn --tta
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import scipy.io as sio
import torch

from sklearn.metrics import classification_report

from src.config import (
    CHECKPOINT_DIR,
    CLASSES,
    DEVICE,
    LABEL_TO_IDX,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TEST_CSV,
    TTA_STRIDE,
    TTA_WINDOW_LENGTH,
)
from src.dataset import get_dataloaders
from src.inference import predict_logits_tta
from src.metrics import evaluate_predictions, get_confusion_matrix
from src.utils import load_checkpoint, plot_confusion_matrix, set_seed
from scripts.train import get_model_instance


def print_confusion_matrix(cm: np.ndarray, class_names=CLASSES) -> None:
    """Print a labeled confusion matrix to the terminal."""
    header = "True\\Pred".ljust(10) + "".join(f"{name:>8}" for name in class_names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = "".join(f"{int(cm[i, j]):>8}" for j in range(len(class_names)))
        print(f"{name:<10}{row}")
    print("-" * len(header))


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
    return metrics, cm, y_true, y_pred


@torch.no_grad()
def evaluate_test_set_tta(
    model,
    test_csv,
    device,
    apply_filter: bool = False,
    window_length: int = TTA_WINDOW_LENGTH,
    stride: int = TTA_STRIDE,
    max_batch_size: int = 32,
):
    """Record-wise multi-crop TTA over the test split CSV."""
    import pandas as pd

    model.eval()
    df = pd.read_csv(test_csv)
    all_preds = []
    all_targets = []

    for _, row in df.iterrows():
        record_id = row["record_id"]
        label_str = row["label"]
        folder_name = record_id[:3]
        file_path = os.path.join(RAW_DATA_DIR, folder_name, f"{record_id}.mat")
        signal = sio.loadmat(file_path)["val"].squeeze().astype(np.float32)

        logits = predict_logits_tta(
            model,
            signal,
            device=device,
            apply_filter=apply_filter,
            window_length=window_length,
            stride=stride,
            max_batch_size=max_batch_size,
        )
        pred = int(torch.argmax(logits).item())
        all_preds.append(pred)
        all_targets.append(LABEL_TO_IDX[label_str])

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    metrics = evaluate_predictions(y_true, y_pred)
    cm = get_confusion_matrix(y_true, y_pred)
    return metrics, cm, y_true, y_pred


def main(args):
    set_seed(42)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    print(f"TTA enabled: {args.tta} | filtering: {args.use_filtering}")

    model = get_model_instance(args.model).to(device)
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR / f"best_{args.model}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)

    print("Running inference on Test Set...")
    if args.tta:
        metrics, cm, y_true, y_pred = evaluate_test_set_tta(
            model,
            test_csv=TEST_CSV,
            device=device,
            apply_filter=args.use_filtering,
            window_length=args.tta_window,
            stride=args.tta_stride,
            max_batch_size=args.batch_size,
        )
        suffix = f"{args.model}_tta"
    else:
        _, _, test_loader = get_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_filtering=args.use_filtering,
            use_augmentation=False,
        )
        metrics, cm, y_true, y_pred = evaluate_test_set(model, test_loader, device)
        suffix = args.model

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 50)
    print(f"TEST EVALUATION RESULTS: {args.model.upper()}" + (" + TTA" if args.tta else ""))
    print("=" * 50)
    print(f"Official Challenge F1 : {metrics['challenge_f1']:.4f}")
    print(f"Macro F1              : {metrics['macro_f1']:.4f}")
    print(f"Overall Accuracy      : {metrics['accuracy'] * 100:.2f}%")
    print("-" * 50)
    print(f"   - F1 Normal (N)       : {metrics['f1_normal_N']:.4f}")
    print(f"   - F1 AF (A)           : {metrics['f1_af_A']:.4f}")
    print(f"   - F1 Other (O)        : {metrics['f1_other_O']:.4f}")
    print(f"   - F1 Noisy (~)        : {metrics['f1_noisy_~']:.4f}")
    print("=" * 50)

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix (rows=True, cols=Predicted):")
    print_confusion_matrix(cm)
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_json_path = RESULTS_DIR / f"{suffix}_test_metrics.json"
    with open(results_json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    report_txt_path = RESULTS_DIR / f"{suffix}_classification_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\nConfusion Matrix (rows=True, cols=Predicted):\n")
        f.write("True\\Pred".ljust(10) + "".join(f"{name:>8}" for name in CLASSES) + "\n")
        for i, name in enumerate(CLASSES):
            row = "".join(f"{int(cm[i, j]):>8}" for j in range(len(CLASSES)))
            f.write(f"{name:<10}{row}\n")

    cm_image_name = f"{suffix}_confusion_matrix.png"
    plot_confusion_matrix(
        cm,
        save_name=cm_image_name,
        title=f"Confusion Matrix - {args.model.upper()}"
        + (" + TTA" if args.tta else "")
        + " (Test Set)",
    )

    print(f"Metrics saved to: {results_json_path}")
    print(f"Classification report saved to: {report_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set.")
    parser.add_argument("--model", type=str, default="crnn", help="Model name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint pt file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size / TTA window batch")
    parser.add_argument("--num_workers", type=int, default=0, help="Num workers")
    parser.add_argument(
        "--use_filtering",
        action="store_true",
        default=False,
        help="Apply software filters",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sliding-window TTA with logit averaging (default: on).",
    )
    parser.add_argument("--tta_window", type=int, default=TTA_WINDOW_LENGTH)
    parser.add_argument("--tta_stride", type=int, default=TTA_STRIDE)

    args = parser.parse_args()
    main(args)
