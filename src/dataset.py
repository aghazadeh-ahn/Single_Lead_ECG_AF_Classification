"""
src/dataset.py
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset
from src.config import (APPLY_FILTERING, BATCH_SIZE, RAW_DATA_DIR, LABEL_TO_IDX, NUM_WORKERS, TARGET_LENGTH, TEST_CSV, TRAIN_CSV, VAL_CSV)
from src.preprocessing import clean_ecg_pipeline

class ECGDataset(Dataset):
    def __init__(self, df_or_path: Union[pd.DataFrame, str, Path], target_length: int = TARGET_LENGTH, apply_filter: bool = APPLY_FILTERING, is_train: bool = True):
        if isinstance(df_or_path, (str, Path)):
            self.df = pd.read_csv(df_or_path)
        else:
            self.df = df_or_path.copy()
        
        self.labels = self.df["label"].tolist()
        self.target_length = target_length
        self.apply_filter = apply_filter
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # اصلاح مسیردهی: استفاده از 3 کاراکتر اول برای پیدا کردن پوشه (مثلاً A01 برای A01706)
        record_id = row["record_id"]
        folder_name = record_id[:3]  
        file_path = os.path.join(RAW_DATA_DIR, folder_name, f"{record_id}.mat")
        
        label_str = row["label"]
        mat_data = sio.loadmat(file_path)
        signal = mat_data["val"].squeeze().astype(np.float32)
        
        signal = clean_ecg_pipeline(signal, apply_filter=self.apply_filter, normalize=True, target_length=self.target_length, is_train=self.is_train)
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        
        label_tensor = torch.tensor(LABEL_TO_IDX[label_str], dtype=torch.long)
        return signal_tensor, label_tensor

def get_dataloaders(train_csv=TRAIN_CSV, val_csv=VAL_CSV, test_csv=TEST_CSV, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, use_filtering=APPLY_FILTERING):
    train_dataset = ECGDataset(train_csv, apply_filter=use_filtering, is_train=True)
    val_dataset = ECGDataset(val_csv, apply_filter=use_filtering, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    test_loader = None
    if test_csv and Path(test_csv).exists():
        test_dataset = ECGDataset(test_csv, apply_filter=use_filtering, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        
    return train_loader, val_loader, test_loader
