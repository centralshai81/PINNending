"""
generate_test1000.py
--------------------
Extracts 1000+ test samples (200 per class x 5 classes = 1000 total)
from your training dataset with high model confidence.

Run: python generate_test1000.py
Output: TEST_1000.csv
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


# ── Config ────────────────────────────────────────────────
SAMPLES_PER_CLASS = 200          # 200 x 5 = 1000 total
OUTPUT_PATH       = 'TEST_1000.csv'
DATASET_PATH      = 'final_data/ieee39_final_dataset.csv'
MODEL_PATH        = 'outputs/checkpoints/best_model.pth'
CHUNK_SIZE        = 5000
# ─────────────────────────────────────────────────────────


def generate_test():
    print("=" * 60)
    print("  Test Dataset Generator  (1000+ samples)")
    print(f"  {SAMPLES_PER_CLASS} samples per class x 5 classes = "
          f"{SAMPLES_PER_CLASS * 5} total")
    print("=" * 60)

    # ── Check files ───────────────────────────────────────
    for path in [DATASET_PATH, MODEL_PATH]:
        if not os.path.exists(path):
            print(f"[ERROR] Not found: {path}")
            return

    # ── Load model ────────────────────────────────────────
    print("\nLoading model...")
    model = PINNMultiClass(in_dim=156, n_bus=39, n_classes=5,
                           hidden=128, depth=3)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print("[OK] Model loaded")

    # ── Column setup ──────────────────────────────────────
    vm_cols   = [f'vm_{i}' for i in range(39)]
    va_cols   = [f'va_{i}' for i in range(39)]
    p_cols    = [f'p_{i}'  for i in range(39)]
    q_cols    = [f'q_{i}'  for i in range(39)]
    feat_cols = vm_cols + va_cols + p_cols + q_cols  # 156

    fault_types = ['Normal', '3LG', 'LG', 'LLG', 'LL']
    label_map   = {'Normal': 0, '3LG': 1, 'LG': 2, 'LLG': 3, 'LL': 4}

    # ── Collect best N samples per class ──────────────────
    # collected[ft] = list of (confidence, row_dict)
    collected = {ft: [] for ft in fault_types}
    total_scanned = 0

    print(f"\nScanning {DATASET_PATH}...")
    print(f"Target: {SAMPLES_PER_CLASS} samples per class\n")

    for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE):
        total_scanned += len(chunk)

        for ft in fault_types:
            if len(collected[ft]) >= SAMPLES_PER_CLASS * 3:
                # Collected enough candidates, skip further scanning for this type
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

            for i, (conf, row) in enumerate(
                    zip(correct_probs, subset.itertuples(index=False))):
                collected[ft].append((float(conf), row._asdict()))

        # Progress
        counts = {ft: len(v) for ft, v in collected.items()}
        done   = sum(1 for ft in fault_types
                     if len(collected[ft]) >= SAMPLES_PER_CLASS * 3)
        print(f"  Scanned {total_scanned:>8,} | "
              f"N:{counts['Normal']:>4} 3LG:{counts['3LG']:>4} "
              f"LG:{counts['LG']:>4} LLG:{counts['LLG']:>4} "
              f"LL:{counts['LL']:>4}",
              end='\r', flush=True)

        if done == 5:
            break

    print(f"\n[OK] Scan complete — {total_scanned:,} rows processed\n")

    # ── Pick top-N by confidence per class ────────────────
    print(f"{'Class':<8} {'Available':<12} {'Selected':<10} {'Min Conf':<10} {'Avg Conf'}")
    print("-" * 52)

    rows_out = []
    for i, ft in enumerate(fault_types):
        candidates = sorted(collected[ft], key=lambda x: -x[0])
        top_n      = candidates[:SAMPLES_PER_CLASS]

        if len(top_n) == 0:
            print(f"  {ft:<8} NOT FOUND — skipping")
            continue

        confs = [c for c, _ in top_n]
        print(f"  {ft:<8} {len(candidates):<12,} {len(top_n):<10} "
              f"{min(confs)*100:.1f}%{'':>4} {sum(confs)/len(confs)*100:.1f}%")

        for rank, (conf, row_dict) in enumerate(top_n):
            out = {
                'time_step' : len(rows_out),
                'label'     : label_map[ft],
                'fault_type': ft,
                'fault_bus' : int(row_dict.get('fault_bus', -1)),
                'fault_Rf'  : float(row_dict.get('fault_Rf', 0.0)),
                'fault_Xf'  : float(row_dict.get('fault_Xf', 0.0)),
                'model_confidence': round(conf, 6),
            }
            for col in feat_cols:
                out[col] = float(row_dict[col])
            rows_out.append(out)

    # ── Shuffle and save ──────────────────────────────────
    print(f"\nShuffling {len(rows_out)} samples...")
    df = pd.DataFrame(rows_out).sample(frac=1, random_state=42).reset_index(drop=True)
    df['time_step'] = range(len(df))   # re-index after shuffle

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Saved: {OUTPUT_PATH}")
    print(f"     Rows    : {len(df):,}")
    print(f"     Columns : {len(df.columns)}")

    # ── Final verification ────────────────────────────────
    print("\nVerification pass...")
    df2   = pd.read_csv(OUTPUT_PATH)
    X2    = torch.tensor(df2[feat_cols].values, dtype=torch.float32)

    all_preds = []
    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(X2), batch_size):
            xb = X2[start:start+batch_size]
            logits, _, _ = model(xb)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.numpy().tolist())

    idx_to_type = ['Normal', '3LG', 'LG', 'LLG', 'LL']
    correct = sum(
        idx_to_type[p] == t
        for p, t in zip(all_preds, df2['fault_type'])
    )
    acc = correct / len(df2)

    print(f"\n{'Class':<8} {'Count':<8} {'Correct':<10} {'Accuracy'}")
    print("-" * 38)
    for ft in fault_types:
        mask    = df2['fault_type'] == ft
        sub_idx = df2.index[mask].tolist()
        n       = len(sub_idx)
        if n == 0: continue
        c = sum(idx_to_type[all_preds[i]] == ft for i in sub_idx)
        print(f"  {ft:<8} {n:<8} {c:<10} {c/n*100:.1f}%")

    print(f"\nOverall Accuracy : {correct}/{len(df2)} = {acc*100:.1f}%")
    print(f"Output file      : {OUTPUT_PATH}")

    if acc >= 0.95:
        print("\n[PASS] TEST_1000.csv is ready for web upload!")
    else:
        print("\n[WARN] Accuracy below 95% — model may need more training.")

    print("=" * 60)


if __name__ == '__main__':
    generate_test()
