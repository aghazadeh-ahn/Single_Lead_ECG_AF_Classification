"""
scripts/prepare_data.py
Generates train, val, and test stratified splits from raw reference files.
Ensures reproducible splits and zero data leakage.

Usage:
    python scripts/prepare_data.py
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    REFERENCE_CSV_PATH,
    SEED,
    SPLITS_DIR,
    TEST_SPLIT_CSV,
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
)
from src.utils import set_seed


def prepare_splits(
    reference_path: Path = REFERENCE_CSV_PATH,
    splits_dir: Path = SPLITS_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> None:
    set_seed(seed)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"📖 Reading reference file from: {reference_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found at {reference_path}")

    df = pd.read_csv(reference_path, names=["record_id", "label"])
    
    # Strip any directory prefixes if present (e.g., A00/A00001 -> A00001)
    df["record_id"] = df["record_id"].apply(lambda x: Path(str(x)).stem)
    
    total_records = len(df)
    print(f"📊 Total records found: {total_records}")
    print("Class distribution in full dataset:")
    print(df["label"].value_counts())

    # Step 1: Split Train vs (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df["label"],
    )

    # Step 2: Split Val vs Test
    val_relative_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative_ratio),
        random_state=seed,
        stratify=temp_df["label"],
    )

    # Sanity checks: Overlap / Leakage
    train_set = set(train_df["record_id"])
    val_set = set(val_df["record_id"])
    test_set = set(test_df["record_id"])

    assert len(train_set.intersection(val_set)) == 0, "❌ Leakage detected between Train and Val!"
    assert len(train_set.intersection(test_set)) == 0, "❌ Leakage detected between Train and Test!"
    assert len(val_set.intersection(test_set)) == 0, "❌ Leakage detected between Val and Test!"
    assert len(train_df) + len(val_df) + len(test_df) == total_records, "❌ Record count mismatch!"

    # Save splits to CSV
    train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(VAL_SPLIT_CSV, index=False)
    test_df.to_csv(TEST_SPLIT_CSV, index=False)

    print("\n✅ Splits created successfully:")
    print(f"   - Train: {len(train_df)} ({len(train_df)/total_records*100:.1f}%) -> {TRAIN_SPLIT_CSV}")
    print(f"   - Val:   {len(val_df)} ({len(val_df)/total_records*100:.1f}%) -> {VAL_SPLIT_CSV}")
    print(f"   - Test:  {len(test_df)} ({len(test_df)/total_records*100:.1f}%) -> {TEST_SPLIT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset splits.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    prepare_splits(seed=args.seed)
