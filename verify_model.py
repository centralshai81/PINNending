"""
verify_model.py
---------------
Tests the trained model on DEMO_PERFECT.csv.
Run after training: python verify_model.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import numpy as np
import pandas as pd
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pinn import PINNMultiClass


def verify():
    print("[OK] Model loaded successfully")

    model = PINNMultiClass(in_dim=156, n_bus=39, n_classes=5, hidden=128, depth=3)
    try:
        model.load_state_dict(
            torch.load('outputs/checkpoints/best_model.pth', weights_only=True)
        )
    except Exception as e:
        print(f"[ERR] Could not load model: {e}")
        return
    model.eval()

    # Load DEMO file
    demo_path = 'DEMO_PERFECT.csv'
    if not os.path.exists(demo_path):
        print(f"[ERR] {demo_path} not found")
        return

    df = pd.read_csv(demo_path)
    print(f"\nTesting on {demo_path} ({len(df)} samples)")

    vm_cols = [f'vm_{i}' for i in range(39)]
    p_cols  = [f'p_{i}'  for i in range(39)]
    q_cols  = [f'q_{i}'  for i in range(39)]

    has_va = 'va_0' in df.columns
    if has_va:
        va_cols      = [f'va_{i}' for i in range(39)]
        feature_cols = vm_cols + va_cols + p_cols + q_cols
        print("Format: 156 features (vm+va+p+q)")
    else:
        feature_cols = vm_cols + p_cols + q_cols
        print("Format: 117 features (no va) — padding with zeros")

    idx_to_type = ['Normal', '3LG', 'LG', 'LLG', 'LL']
    true_labels = df['fault_type'].values if 'fault_type' in df.columns \
                  else ['Unknown'] * len(df)

    print(f"\n{'Sample':<8} {'True':<8} {'Predicted':<12} {'Confidence':<12} {'Result'}")
    print("-" * 55)

    correct = 0
    for i in range(len(df)):
        if has_va:
            feat = df[feature_cols].iloc[i].values.astype(np.float32)
        else:
            vm = df[vm_cols].iloc[i].values.astype(np.float32)
            va = np.zeros(39, dtype=np.float32)
            p  = df[p_cols].iloc[i].values.astype(np.float32)
            q  = df[q_cols].iloc[i].values.astype(np.float32)
            feat = np.concatenate([vm, va, p, q])

        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = model(x)
            probs        = torch.softmax(logits, dim=1).numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred     = idx_to_type[pred_idx]
        conf     = float(probs[pred_idx])
        ok       = (pred == true_labels[i])
        if ok:
            correct += 1
        mark = "[OK]" if ok else "[X]"

        print(f"{i+1:<8} {true_labels[i]:<8} {pred:<12} {conf*100:.1f}%{'':<6} {mark}")

    acc = correct / len(df)
    print(f"\nAccuracy: {correct}/{len(df)} = {acc*100:.0f}%")

    if acc >= 1.0:
        print("PASS — all samples correct")
    elif acc >= 0.8:
        print("PARTIAL PASS — some misclassifications, retrain may help")
    else:
        print("FAIL — accuracy too low, check training")


if __name__ == '__main__':
    verify()
