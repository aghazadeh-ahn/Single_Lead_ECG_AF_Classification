import argparse, json, sys, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter # 1. اضافه کردن ایمپورت

# اصلاح import ها برای دسترسی صحیح
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import BATCH_SIZE, CHECKPOINT_DIR, CLASSES, DEVICE, EPOCHS, LEARNING_RATE, LOGS_DIR, SEED, WEIGHT_DECAY, LABEL_TO_IDX
from src.dataset import get_dataloaders
from src.metrics import compute_challenge_f1, evaluate_predictions
from src.utils import save_checkpoint, set_seed

def get_model_instance(model_name: str, num_classes: int = len(CLASSES)) -> nn.Module:
    model_name = model_name.lower()
    if model_name in ["baseline", "baseline_cnn", "cnn"]:
        from src.models.baseline_cnn import Baseline1DCNN
        return Baseline1DCNN(num_classes=num_classes)

    if model_name in ["resnet", "resnet1d"]:
        from src.models.resnet1d import ResNet1D
        return ResNet1D(num_classes=num_classes)

    raise ValueError(f"Unknown model name: '{model_name}'")

def compute_class_weights(train_dataset, device):
    labels = train_dataset.labels 
    indices = [LABEL_TO_IDX[l] for l in labels]
    class_counts = np.bincount(indices, minlength=len(CLASSES))
    total_samples = len(indices)
    class_weights = total_samples / (len(CLASSES) * (class_counts + 1e-6))
    return torch.tensor(class_weights, dtype=torch.float32).to(device)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, all_preds, all_targets = 0.0, [], []
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += loss.item() * signals.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())
    return running_loss / len(loader.dataset), compute_challenge_f1(np.array(all_targets), np.array(all_preds))

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, all_preds, all_targets = 0.0, [], []
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        logits = model(signals)
        loss = criterion(logits, labels)
        running_loss += loss.item() * signals.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())
    val_loss = running_loss / len(loader.dataset)
    metrics = evaluate_predictions(np.array(all_targets), np.array(all_preds))
    metrics["loss"] = float(val_loss)
    return metrics

def main(args):
    set_seed(args.seed)
    device = torch.device(DEVICE)
    
    # 2. ایجاد دایرکتوری لاگ و اینیت کردن نویسنده تنسوربورد
    log_dir = Path(LOGS_DIR) / f"{args.model}_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, use_filtering=args.use_filtering)
    model = get_model_instance(args.model).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{args.model}' loaded. Params: {total_params:,}")
    
    if args.weighted_loss:
        class_weights = compute_class_weights(train_loader.dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_f1 = -1.0
    
    try: # برای اینکه اگر وسط کار متوقف شد هم writer.close() اجرا شود
        for epoch in range(1, args.epochs + 1):
            train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = validate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val F1: {val_metrics['challenge_f1']:.4f}")
            
            # 3. ثبت لاگ در TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('F1/train', train_f1, epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('F1/val', val_metrics['challenge_f1'], epoch)

            if val_metrics['challenge_f1'] > best_val_f1:
                best_val_f1 = val_metrics['challenge_f1']
                save_checkpoint(model, optimizer, epoch, val_metrics['loss'], best_val_f1, f"best_{args.model}.pt")
    
    finally:
        writer.close() # 4. بستن نویسنده
        print("Training finished. Writer closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weighted_loss", action="store_true", default=True)
    parser.add_argument("--use_filtering", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
