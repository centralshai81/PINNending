"""
generate_demo.py  (v3 - extracts from training data)
-----------------------------------------------------
Extracts one high-confidence sample per class from your training
dataset. This guarantees demo samples look exactly like training
data — no Ybus mismatch, no simulation errors.

Run: python generate_demo.py
Output: DEMO_PERFECT.csv
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pinn import PINNMultiClass


def generate_demo():
    dataset_path = 'final_data/ieee39_final_dataset.csv'
    model_path   = 'outputs/checkpoints/best_model.pth'
    output_path  = 'DEMO_PERFECT.csv'

    # Check files exist
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Not found: {dataset_path}")
        print("  Run: python generate_dataset.py")
        return
    if not os.path.exists(model_path):
        print(f"[ERROR] Not found: {model_path}")
        print("  Run: python train_pinn.py")
        return

    # Load model
    print("Loading model...")
    model = PINNMultiClass(in_dim=156, n_bus=39, n_classes=5, hidden=128, depth=3)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("[OK] Model loaded")

    # Check dataset format
    header = pd.read_csv(dataset_path, nrows=1)
    if 'va_0' not in header.columns:
        print("[ERROR] Dataset missing va columns.")
        print("  Run: python generate_dataset.py")
        return

    vm_cols   = [f'vm_{i}' for i in range(39)]
    va_cols   = [f'va_{i}' for i in range(39)]
    p_cols    = [f'p_{i}'  for i in range(39)]
    q_cols    = [f'q_{i}'  for i in range(39)]
    feat_cols = vm_cols + va_cols + p_cols + q_cols  # 156

    fault_types  = ['Normal', '3LG', 'LG', 'LLG', 'LL']
    label_map    = {'Normal': 0, '3LG': 1, 'LG': 2, 'LLG': 3, 'LL': 4}
    idx_to_type  = ['Normal', '3LG', 'LG', 'LLG', 'LL']

    print(f"\nScanning {dataset_path} for best samples...")
    print("  Looking for samples each fault type classifies with highest confidence\n")

    # best[ft] = (confidence, row_as_dict)
    best = {ft: (0.0, None) for ft in fault_types}

    chunk_size = 5000
    total_scanned = 0

    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        total_scanned += len(chunk)

        for ft in fault_types:
            # Already have a near-perfect sample for this type
            if best[ft][0] >= 0.97:
                continue

            subset = chunk[chunk['fault_type'] == ft]
            if len(subset) == 0:
                continue

            X = torch.tensor(subset[feat_cols].values, dtype=torch.float32)
            with torch.no_grad():
                logits, _, _ = model(X)
                probs = torch.softmax(logits, dim=1).numpy()

            correct_idx   = label_map[ft]
            correct_probs = probs[:, correct_idx]
            best_i        = int(np.argmax(correct_probs))
            best_conf     = float(correct_probs[best_i])

            if best_conf > best[ft][0]:
                best[ft] = (best_conf, subset.iloc[best_i].to_dict())

        # Print progress
        found = sum(1 for ft in fault_types if best[ft][1] is not None)
        confs = [f"{best[ft][0]*100:.0f}%" for ft in fault_types]
        print(f"  Scanned {total_scanned:>7,} rows | "
              f"found {found}/5 | {' '.join(confs)}",
              end='\r', flush=True)

        # Stop when all types have >= 95% confidence
        if all(best[ft][0] >= 0.95 for ft in fault_types):
            print(f"\n[OK] All 5 types found with >= 95% confidence")
            break

    print(f"\n\nResults:")
    print(f"{'Type':<8} {'Best Conf':<12} {'Fault Bus':<12} {'Min Vm':<10}")
    print("-" * 44)

    rows = []
    for i, ft in enumerate(fault_types):
        conf, row_dict = best[ft]
        if row_dict is None:
            print(f"  {ft:<8} NOT FOUND")
            continue

        vm_vals = np.array([row_dict[f'vm_{j}'] for j in range(39)])
        fb      = int(row_dict.get('fault_bus', -1))
        print(f"  {ft:<8} {conf*100:.1f}%       {fb:<12} {vm_vals.min():.4f}")

        out = {
            'time_step' : i,
            'label'     : label_map[ft],
            'fault_type': ft,
            'fault_bus' : fb,
            'fault_Rf'  : float(row_dict.get('fault_Rf', 0.01)),
            'fault_Xf'  : float(row_dict.get('fault_Xf', 0.01)),
        }
        for col in feat_cols:
            out[col] = float(row_dict[col])
        rows.append(out)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Saved: {output_path}")
    print(f"     Rows    : {len(df)}")
    print(f"     Columns : {len(df.columns)}")

    # Verify saved file
    print("\nVerification on saved file:")
    df2 = pd.read_csv(output_path)
    X2  = torch.tensor(df2[feat_cols].values, dtype=torch.float32)
    with torch.no_grad():
        logits2, _, _ = model(X2)
        probs2 = torch.softmax(logits2, dim=1).numpy()

    print(f"\n{'#':<4} {'True':<8} {'Predicted':<12} {'Confidence':<12} {'Result'}")
    print("-" * 50)
    correct = 0
    for i, row in df2.iterrows():
        true_ft  = row['fault_type']
        pred_idx = int(np.argmax(probs2[i]))
        pred_ft  = idx_to_type[pred_idx]
        conf     = float(probs2[i][pred_idx])
        ok       = pred_ft == true_ft
        if ok:
            correct += 1
        mark = "[OK]" if ok else "[X]"
        print(f"  {i+1:<3} {true_ft:<8} {pred_ft:<12} {conf*100:.1f}%       {mark}")

    acc = correct / len(df2)
    print(f"\nAccuracy: {correct}/{len(df2)} = {acc*100:.0f}%")

    if acc == 1.0:
        print("[PASS] DEMO_PERFECT.csv is ready")
    else:
        print("[WARN] Model accuracy issue — not a data problem.")
        print("       The model may need more training.")


if __name__ == '__main__':
    generate_demo()
