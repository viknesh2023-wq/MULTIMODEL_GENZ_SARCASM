"""
Dataset Verification & Reconstruction Utility
Project: Gen-Z Multimodal Sarcasm Detection System (Team 14)

Verifies all datasets and reconstructs text_sarcasm_combined.csv from train, validation, and test splits if needed.
"""

import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "FINAL_DATASETS" / "TEXT_SARCASM" / "SPLIT"
COMBINED_FILE = PROJECT_ROOT / "FINAL_DATASETS" / "TEXT_SARCASM" / "text_sarcasm_combined.csv"

def verify_and_reconstruct():
    print("=" * 70)
    print("Verifying Final Datasets...")
    print("=" * 70)
    
    train_file = SPLIT_DIR / "train.csv"
    val_file = SPLIT_DIR / "validation.csv"
    test_file = SPLIT_DIR / "test.csv"
    
    for f in [train_file, val_file, test_file]:
        if f.exists():
            print(f"[OK] Split found: {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"[MISSING] Split missing: {f}")
            return
            
    if COMBINED_FILE.exists():
        print(f"[OK] text_sarcasm_combined.csv already exists ({COMBINED_FILE.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print("[INFO] Reconstructing text_sarcasm_combined.csv from splits...")
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        df_test = pd.read_csv(test_file)
        df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)
        df_combined.to_csv(COMBINED_FILE, index=False)
        print(f"[SUCCESS] Reconstructed text_sarcasm_combined.csv ({len(df_combined):,} rows)")
        
    print("=" * 70)

if __name__ == "__main__":
    verify_and_reconstruct()
