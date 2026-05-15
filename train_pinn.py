"""
train_pinn.py  (with progress bars and hardware detection)
----------------------------------------------------------
Two-phase PINN training.
Shows: hardware info, ETA per epoch, batch progress bar, live loss values.

Requirements:
    pip install tqdm
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARN] tqdm not installed. Run: pip install tqdm")

from models.pinn import PINNMultiClass
from utils import build_Ybus, physics_loss_fn


def detect_hardware():
    print("=" * 60)
    print("Hardware Detection")
    print("=" * 60)
    if torch.cuda.is_available():
        device   = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name}  ({gpu_mem:.1f} GB)")
        print(f"      Estimated total time: ~30-45 minutes")
    else:
        device = torch.device('cpu')
        import os as _os
        cores = _os.cpu_count()
        torch.set_num_threads(cores)
        print(f"[CPU] {cores} cores  (all threads enabled)")
        print(f"      Estimated total time: 2-4 hours")
        print(f"      Tip: use n_per_class=60000 in generate_dataset.py")
        print(f"           to reduce to ~1-2 hours")
    print("=" * 60)
    return device


class FaultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):      return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def load_ybus(path='final_data/ieee39_physics.npz'):
    data = np.load(path)
    return torch.tensor(build_Ybus(39, data['IEEE39_BRANCH']), dtype=torch.complex64)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_v, y_v in loader:
            X_v, y_v = X_v.to(device), y_v.to(device)
            preds    = torch.argmax(model(X_v)[0], dim=1)
            correct += (preds == y_v).sum().item()
            total   += len(y_v)
    return correct / total


def run_epoch(model, loader, criterion, optimizer,
              Ybus_tensor, lambda_physics, device, label):
    """One training epoch with tqdm batch progress bar."""
    model.train()
    sum_cls = sum_phys = 0.0
    n_correct = n_total = 0

    if TQDM_AVAILABLE:
        bar = tqdm(loader, desc=label, ncols=110, leave=False,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                              '[{elapsed}<{remaining}] {postfix}')
    else:
        bar = loader

    for i, (X_b, y_b) in enumerate(bar):
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()

        logits, Vm_hat, Va_hat = model(X_b)
        cls_loss = criterion(logits, y_b)

        if lambda_physics > 0:
            P_meas    = X_b[:, 78:117].float()
            Q_meas    = X_b[:, 117:156].float()
            phys_loss = physics_loss_fn(
                Vm_hat, Va_hat, P_meas, Q_meas,
                Ybus_tensor, lambda_physics=lambda_physics
            )
            loss      = cls_loss + phys_loss
            sum_phys += phys_loss.item()
        else:
            loss = cls_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sum_cls   += cls_loss.item()
        preds      = torch.argmax(logits, dim=1)
        n_correct += (preds == y_b).sum().item()
        n_total   += len(y_b)

        if TQDM_AVAILABLE and i % 10 == 0:
            if lambda_physics > 0:
                bar.set_postfix(
                    cls=f"{sum_cls/(i+1):.4f}",
                    phys=f"{sum_phys/(i+1):.5f}",
                    acc=f"{n_correct/n_total:.3f}"
                )
            else:
                bar.set_postfix(
                    cls=f"{sum_cls/(i+1):.4f}",
                    acc=f"{n_correct/n_total:.3f}"
                )

    n = len(loader)
    return sum_cls/n, sum_phys/n if lambda_physics > 0 else 0.0, n_correct/n_total


def train():
    device = detect_hardware()

    print("\n" + "=" * 60)
    print("PINN Fault Diagnosis — Two-Phase Training")
    print("  in_dim=156  (vm + va + p + q)")
    print("=" * 60)

    os.makedirs('outputs/checkpoints', exist_ok=True)

    # ── Load data
    data_path = 'final_data/ieee39_final_dataset.csv'
    print(f"\nLoading {data_path} ...")
    t0 = time.time()
    df = pd.read_csv(data_path)

    if 'va_0' not in df.columns:
        print("\n[ERROR] va_0 column missing.")
        print("  Run: python generate_dataset.py  (uses new version with angles)")
        return

    fault_type_map = {'Normal': 0, '3LG': 1, 'LG': 2, 'LLG': 3, 'LL': 4}
    vm_cols = [f'vm_{i}' for i in range(39)]
    va_cols = [f'va_{i}' for i in range(39)]
    p_cols  = [f'p_{i}'  for i in range(39)]
    q_cols  = [f'q_{i}'  for i in range(39)]
    fcols   = vm_cols + va_cols + p_cols + q_cols   # 156

    X_all = df[fcols].values.astype(np.float32)
    y_all = df['fault_type'].map(fault_type_map).values.astype(np.int64)
    print(f"[OK] Loaded in {time.time()-t0:.1f}s")
    print(f"     Samples : {len(X_all):,}")
    print(f"     Features: {X_all.shape[1]}  (should be 156)")
    print(f"     Classes : {np.bincount(y_all)}")

    # ── Split
    np.random.seed(42)
    idx   = np.random.permutation(len(X_all))
    n_val = int(0.1 * len(X_all))
    v_idx = idx[:n_val]
    t_idx = idx[n_val:]

    BATCH = 1024
    train_loader = DataLoader(FaultDataset(X_all[t_idx], y_all[t_idx]),
                              batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(FaultDataset(X_all[v_idx], y_all[v_idx]),
                              batch_size=2048,  shuffle=False, num_workers=0)
    print(f"\n     Train   : {len(t_idx):,}  |  Val: {len(v_idx):,}")
    print(f"     Batches : {len(train_loader)} per epoch")

    # ── Ybus
    Ybus_tensor = load_ybus().to(device)
    print("[OK] Ybus ready")

    # ── Model
    model    = PINNMultiClass(in_dim=156, n_bus=39, n_classes=5,
                              hidden=128, depth=3).to(device)
    criterion = nn.CrossEntropyLoss()
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model ready  ({n_params:,} parameters)")

    best_val_acc = 0.0
    t_train_start = time.time()

    # ══════════════════════════════════════════════════════════
    # PHASE 1  —  Classification only
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: Classification only")
    print("=" * 60)

    opt1   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=5,
                                                   factor=0.5, min_lr=1e-5)
    P1_EPOCHS   = 80
    P1_PATIENCE = 15
    p1_wait     = 0
    t1 = time.time()

    for epoch in range(P1_EPOCHS):
        label = f"P1 E{epoch+1:3d}/{P1_EPOCHS}"
        avg_cls, _, train_acc = run_epoch(
            model, train_loader, criterion, opt1,
            Ybus_tensor, lambda_physics=0.0, device=device, label=label
        )
        val_acc = evaluate(model, val_loader, device)
        sched1.step(avg_cls)

        elapsed = time.time() - t1
        eta     = elapsed / (epoch + 1) * (P1_EPOCHS - epoch - 1)
        lr_now  = opt1.param_groups[0]['lr']

        print(f"  Phase1 Epoch {epoch+1:3d}/{P1_EPOCHS} | "
              f"Loss: {avg_cls:.4f} | "
              f"Train: {train_acc:.4f} | "
              f"Val: {val_acc:.4f} | "
              f"lr: {lr_now:.2e} | "
              f"ETA: {eta/60:.1f}min")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            p1_wait = 0
            torch.save(model.state_dict(), 'outputs/checkpoints/best_model.pth')
            print(f"         [SAVED] val acc: {best_val_acc:.4f}")
        else:
            p1_wait += 1
            if p1_wait >= P1_PATIENCE:
                print(f"\n  Early stop (no improvement for {P1_PATIENCE} epochs)")
                break

    p1_best = best_val_acc
    t1_done = time.time() - t1
    print(f"\nPhase 1 done in {t1_done/60:.1f} min  |  Best val: {p1_best:.4f}")

    # Reload best
    model.load_state_dict(
        torch.load('outputs/checkpoints/best_model.pth', weights_only=True)
    )
    model = model.to(device)

    # ══════════════════════════════════════════════════════════
    # PHASE 2  —  Classification + Physics
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: Classification + Physics loss")
    print("  Lambda warmup: 0.0001 -> 0.001")
    print("=" * 60)

    opt2   = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=8,
                                                   factor=0.5, min_lr=1e-6)
    P2_EPOCHS   = 100
    P2_PATIENCE = 20
    p2_wait     = 0
    t2 = time.time()

    for epoch in range(P2_EPOCHS):
        lam   = 0.0001 + 0.0009 * (epoch / P2_EPOCHS)
        label = f"P2 E{epoch+1:3d}/{P2_EPOCHS} lam={lam:.5f}"

        avg_cls, avg_phys, train_acc = run_epoch(
            model, train_loader, criterion, opt2,
            Ybus_tensor, lambda_physics=lam, device=device, label=label
        )
        val_acc = evaluate(model, val_loader, device)
        sched2.step(avg_cls)

        elapsed = time.time() - t2
        eta     = elapsed / (epoch + 1) * (P2_EPOCHS - epoch - 1)
        lr_now  = opt2.param_groups[0]['lr']

        print(f"  Phase2 Epoch {epoch+1:3d}/{P2_EPOCHS} | "
              f"Cls: {avg_cls:.4f} | "
              f"Phys: {avg_phys:.5f} | "
              f"lam: {lam:.5f} | "
              f"Train: {train_acc:.4f} | "
              f"Val: {val_acc:.4f} | "
              f"lr: {lr_now:.2e} | "
              f"ETA: {eta/60:.1f}min")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            p2_wait = 0
            torch.save(model.state_dict(), 'outputs/checkpoints/best_model.pth')
            print(f"         [SAVED] val acc: {best_val_acc:.4f}")
        else:
            p2_wait += 1
            if p2_wait >= P2_PATIENCE:
                print(f"\n  Early stop (no improvement for {P2_PATIENCE} epochs)")
                break

    total_time = time.time() - t_train_start

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Phase 1 time       : {t1_done/60:.1f} min")
    print(f"  Phase 2 time       : {(time.time()-t2)/60:.1f} min")
    print(f"  Total time         : {total_time/60:.1f} min")
    print(f"  Phase 1 best val   : {p1_best:.4f}")
    print(f"  Final best val     : {best_val_acc:.4f}")
    print(f"  Physics gain       : +{(best_val_acc-p1_best)*100:.2f}%")
    print(f"  Saved to           : outputs/checkpoints/best_model.pth")
    print("=" * 60)
    print("\nRun: python verify_model.py")


if __name__ == '__main__':
    train()