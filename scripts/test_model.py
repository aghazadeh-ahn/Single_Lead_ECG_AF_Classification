import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. تنظیم مسیر اصلی پروژه برای پیدا کردن ماژول ها ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- 2. ایمپورت ماژول های پروژه ---
from src.dataset import get_dataloaders
from src.models.baseline_cnn import Baseline1DCNN
from src.metrics import evaluate_predictions
from src.config import CLASSES, DEVICE, CHECKPOINT_DIR

def run_test():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # لود کردن دیتاست تست
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=0)
    
    # ساخت معماری مدل
    model = Baseline1DCNN(num_classes=len(CLASSES)).to(device)
    checkpoint_path = Path(CHECKPOINT_DIR) / "best_baseline.pt"
    
    # بررسی وجود فایل
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    # --- 3. لود صحیح وزن ها (اصلاح شده) ---
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # اینجا کلید 'model_state_dict' را از دیکشنری چک‌پوینت استخراج می‌کنیم
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded weights from epoch {checkpoint.get('epoch', 'Unknown')}")
    else:
        # اگر فایل مستقیماً دیکشنری وزن ها باشد (که احتمالش کم است)
        model.load_state_dict(checkpoint)
        print("Direct weights loaded.")
    
    model.eval()

    # --- 4. اجرای ارزیابی ---
    all_preds, all_targets = [], []
    print("Running evaluation on test set...")
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            logits = model(signals)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    # --- 5. نمایش نتایج ---
    metrics = evaluate_predictions(np.array(all_targets), np.array(all_preds))
    print("\n--- Final Test Set Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # --- 6. ذخیره Confusion Matrix ---
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    run_test()
