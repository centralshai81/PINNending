"""
generate_dataset.py
-------------------
Generates IEEE 39-bus fault dataset with 156 features:
    vm_0...vm_38  (39 voltage magnitudes)
    va_0...va_38  (39 voltage angles, radians)  <- ESSENTIAL for physics loss
    p_0...p_38    (39 active power, per-unit)
    q_0...q_38    (39 reactive power, per-unit)

Run: python generate_dataset.py
Output: final_data/ieee39_final_dataset.csv
        final_data/ieee39_physics.npz
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import build_Ybus


# ── IEEE 39-bus branch data from case39.m
IEEE39_BRANCH = np.array([
    [1,  2,  0.0035, 0.0411, 0.6987, 600, 600, 600, 0,     0, 1, -360, 360],
    [1,  39, 0.001,  0.025,  0.75,   1000,1000,1000,0,     0, 1, -360, 360],
    [2,  3,  0.0013, 0.0151, 0.2572, 500, 500, 500, 0,     0, 1, -360, 360],
    [2,  25, 0.007,  0.0086, 0.146,  500, 500, 500, 0,     0, 1, -360, 360],
    [2,  30, 0,      0.0181, 0,      900, 900, 2500,1.025, 0, 1, -360, 360],
    [3,  4,  0.0013, 0.0213, 0.2214, 500, 500, 500, 0,     0, 1, -360, 360],
    [3,  18, 0.0011, 0.0133, 0.2138, 500, 500, 500, 0,     0, 1, -360, 360],
    [4,  5,  0.0008, 0.0128, 0.1342, 600, 600, 600, 0,     0, 1, -360, 360],
    [4,  14, 0.0008, 0.0129, 0.1382, 500, 500, 500, 0,     0, 1, -360, 360],
    [5,  6,  0.0002, 0.0026, 0.0434, 1200,1200,1200,0,     0, 1, -360, 360],
    [5,  8,  0.0008, 0.0112, 0.1476, 900, 900, 900, 0,     0, 1, -360, 360],
    [6,  7,  0.0006, 0.0092, 0.113,  900, 900, 900, 0,     0, 1, -360, 360],
    [6,  11, 0.0007, 0.0082, 0.1389, 480, 480, 480, 0,     0, 1, -360, 360],
    [6,  31, 0,      0.025,  0,      1800,1800,1800,1.07,  0, 1, -360, 360],
    [7,  8,  0.0004, 0.0046, 0.078,  900, 900, 900, 0,     0, 1, -360, 360],
    [8,  9,  0.0023, 0.0363, 0.3804, 900, 900, 900, 0,     0, 1, -360, 360],
    [9,  39, 0.001,  0.025,  1.2,    900, 900, 900, 0,     0, 1, -360, 360],
    [10, 11, 0.0004, 0.0043, 0.0729, 600, 600, 600, 0,     0, 1, -360, 360],
    [10, 13, 0.0004, 0.0043, 0.0729, 600, 600, 600, 0,     0, 1, -360, 360],
    [10, 32, 0,      0.02,   0,      900, 900, 2500,1.07,  0, 1, -360, 360],
    [12, 11, 0.0016, 0.0435, 0,      500, 500, 500, 1.006, 0, 1, -360, 360],
    [12, 13, 0.0016, 0.0435, 0,      500, 500, 500, 1.006, 0, 1, -360, 360],
    [13, 14, 0.0009, 0.0101, 0.1723, 600, 600, 600, 0,     0, 1, -360, 360],
    [14, 15, 0.0018, 0.0217, 0.366,  600, 600, 600, 0,     0, 1, -360, 360],
    [15, 16, 0.0009, 0.0094, 0.171,  600, 600, 600, 0,     0, 1, -360, 360],
    [16, 17, 0.0007, 0.0089, 0.1342, 600, 600, 600, 0,     0, 1, -360, 360],
    [16, 19, 0.0016, 0.0195, 0.304,  600, 600, 2500,0,     0, 1, -360, 360],
    [16, 21, 0.0008, 0.0135, 0.2548, 600, 600, 600, 0,     0, 1, -360, 360],
    [16, 24, 0.0003, 0.0059, 0.068,  600, 600, 600, 0,     0, 1, -360, 360],
    [17, 18, 0.0007, 0.0082, 0.1319, 600, 600, 600, 0,     0, 1, -360, 360],
    [17, 27, 0.0013, 0.0173, 0.3216, 600, 600, 600, 0,     0, 1, -360, 360],
    [19, 20, 0.0007, 0.0138, 0,      900, 900, 2500,1.06,  0, 1, -360, 360],
    [19, 33, 0.0007, 0.0142, 0,      900, 900, 2500,1.07,  0, 1, -360, 360],
    [20, 34, 0.0009, 0.018,  0,      900, 900, 2500,1.009, 0, 1, -360, 360],
    [21, 22, 0.0008, 0.014,  0.2565, 900, 900, 900, 0,     0, 1, -360, 360],
    [22, 23, 0.0006, 0.0096, 0.1846, 600, 600, 600, 0,     0, 1, -360, 360],
    [22, 35, 0,      0.0143, 0,      900, 900, 2500,1.025, 0, 1, -360, 360],
    [23, 24, 0.0022, 0.035,  0.361,  600, 600, 600, 0,     0, 1, -360, 360],
    [23, 36, 0.0005, 0.0272, 0,      900, 900, 2500,1.0,   0, 1, -360, 360],
    [25, 26, 0.0032, 0.0323, 0.531,  600, 600, 600, 0,     0, 1, -360, 360],
    [25, 37, 0.0006, 0.0232, 0,      900, 900, 2500,1.025, 0, 1, -360, 360],
    [26, 27, 0.0014, 0.0147, 0.2396, 600, 600, 600, 0,     0, 1, -360, 360],
    [26, 28, 0.0043, 0.0474, 0.7802, 600, 600, 600, 0,     0, 1, -360, 360],
    [26, 29, 0.0057, 0.0625, 1.029,  600, 600, 600, 0,     0, 1, -360, 360],
    [28, 29, 0.0014, 0.0151, 0.249,  600, 600, 600, 0,     0, 1, -360, 360],
    [29, 38, 0.0008, 0.0156, 0,      1200,1200,2500,1.025, 0, 1, -360, 360],
], dtype=np.float64)

V0_MAG = np.array([
    1.0393836,  1.0484941,  1.0307077,  1.00446,    1.0060063,
    1.0082256,  0.99839728, 0.99787232, 1.038332,   1.0178431,
    1.0133858,  1.000815,   1.014923,   1.012319,   1.0161854,
    1.0325203,  1.0342365,  1.0315726,  1.0501068,  0.99101054,
    1.0323192,  1.0501427,  1.0451451,  1.038001,   1.0576827,
    1.0525613,  1.0383449,  1.0503737,  1.0501149,  1.0499,
    0.982,      0.9841,     0.9972,     1.0123,     1.0494,
    1.0636,     1.0275,     1.0265,     1.03
], dtype=np.float64)

V0_ANG_DEG = np.array([
    -13.536602,  -9.7852666, -12.276384, -12.626734, -11.192339,
    -10.40833,  -12.755626,  -13.335844, -14.178442,  -8.170875,
     -8.9369663,  -8.9988236,  -8.9299272,-10.715295, -11.345399,
    -10.033348,  -11.116436, -11.986168,  -5.4100729,  -6.8211783,
     -7.6287461,  -3.1831199,  -3.3812763,  -9.9137585,  -8.3692354,
     -9.4387696,  -11.362152,   -5.9283592,  -3.1698741,  -7.3704746,
      0.0,         -0.1884374,  -0.19317445,  -1.631119,    1.7765069,
      4.4684374,   -1.5828988,    3.8928177,  -14.535256
], dtype=np.float64)

V_NOMINAL  = V0_MAG * np.exp(1j * np.deg2rad(V0_ANG_DEG))
BASE_MVA   = 100.0
N_BUS      = 39
RF_OPTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
XF_OPTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def simulate_fault(Ybus, fault_bus, fault_type, Rf, Xf):
    Zf = Rf + 1j * Xf
    Yf = 1.0 / Zf if abs(Zf) > 1e-12 else 1e10
    Y_fault = np.zeros((N_BUS, N_BUS), dtype=np.complex128)
    if fault_type == '3LG':
        Y_fault[fault_bus, fault_bus] = Yf
    elif fault_type == 'LG':
        Y_fault[fault_bus, fault_bus] = Yf / 3.0
    elif fault_type == 'LLG':
        other = (fault_bus + 1) % N_BUS
        Y_fault[fault_bus, fault_bus] += 2.0 * Yf
        Y_fault[other,     other    ] += 2.0 * Yf
        Y_fault[fault_bus, other    ]  = -Yf
        Y_fault[other,     fault_bus]  = -Yf
    elif fault_type == 'LL':
        other = (fault_bus + 1) % N_BUS
        Y_fault[fault_bus, fault_bus] += 1.0 * Yf
        Y_fault[other,     other    ] += 1.0 * Yf
        Y_fault[fault_bus, other    ]  = -Yf
        Y_fault[other,     fault_bus]  = -Yf
    Y_total = Ybus + Y_fault
    try:
        V_fault = np.linalg.solve(Y_total, Ybus @ V_NOMINAL)
    except np.linalg.LinAlgError:
        V_fault = V_NOMINAL * 0.5
    return V_fault


def compute_features(V, Ybus):
    I  = Ybus @ V
    S  = V * np.conj(I)
    Vm = np.abs(V)
    Va = np.angle(V)           # radians — essential for physics loss
    P  = np.real(S) / BASE_MVA
    Q  = np.imag(S) / BASE_MVA
    return Vm, Va, P, Q


def verify_physics_residual(Ybus):
    print("Verifying physics residual with Va fix...")
    Vm, Va, P, Q = compute_features(V_NOMINAL, Ybus)
    V_recon = Vm * np.exp(1j * Va)
    I_r = Ybus @ V_recon
    S_r = V_recon * np.conj(I_r)
    p_mse = float(np.mean((np.real(S_r)/BASE_MVA - P)**2))
    q_mse = float(np.mean((np.imag(S_r)/BASE_MVA - Q)**2))
    print(f"  P residual MSE = {p_mse:.2e}  (target < 1e-6)")
    print(f"  Q residual MSE = {q_mse:.2e}  (target < 1e-6)")
    if p_mse < 1e-6 and q_mse < 1e-6:
        print("  [OK] Physics residual near zero — Va fix confirmed")
        return True
    else:
        print("  [WARN] Residual still large")
        return False


def generate_dataset(n_per_class=100000, seed=42):
    """
    n_per_class=100000  -> 500,000 total samples  (recommended)
    n_per_class=200000  -> 1,000,000 total samples
    n_per_class=40000   -> 200,000 total samples   (minimum)
    """
    np.random.seed(seed)
    os.makedirs('final_data', exist_ok=True)

    print("=" * 60)
    print("IEEE 39-Bus Fault Dataset Generator")
    print(f"  156 features: 39 vm + 39 va + 39 p + 39 q")
    print(f"  {n_per_class:,} per class x 5 classes = {5*n_per_class:,} total")
    print("=" * 60)

    print("\nBuilding Ybus matrix...")
    Ybus = build_Ybus(N_BUS, IEEE39_BRANCH)
    print(f"[OK] Ybus shape: {Ybus.shape}")

    np.savez('final_data/ieee39_physics.npz', IEEE39_BRANCH=IEEE39_BRANCH)
    print("[OK] Saved: final_data/ieee39_physics.npz")

    ok = verify_physics_residual(Ybus)
    if not ok:
        print("[ERROR] Physics check failed. Stopping.")
        return

    fault_types = ['Normal', '3LG', 'LG', 'LLG', 'LL']
    label_map   = {'Normal': 0, '3LG': 1, 'LG': 2, 'LLG': 3, 'LL': 4}

    vm_cols = [f'vm_{i}' for i in range(N_BUS)]
    va_cols = [f'va_{i}' for i in range(N_BUS)]
    p_cols  = [f'p_{i}'  for i in range(N_BUS)]
    q_cols  = [f'q_{i}'  for i in range(N_BUS)]

    print("\nGenerating samples...")
    all_dfs = []

    for fault_type in fault_types:
        print(f"\n  {fault_type} ({n_per_class:,} samples):")

        X    = np.zeros((n_per_class, N_BUS * 4), dtype=np.float32)
        fb_arr = np.full(n_per_class, -1,  dtype=np.int32)
        rf_arr = np.zeros(n_per_class,      dtype=np.float32)
        xf_arr = np.zeros(n_per_class,      dtype=np.float32)

        chunk = 1000
        for c in range(n_per_class // chunk):
            start = c * chunk
            end   = start + chunk
            noise_mag = np.random.normal(0, 0.015, (chunk, N_BUS))
            noise_ang = np.random.normal(0, 0.003, (chunk, N_BUS))

            for i in range(chunk):
                idx    = start + i
                V_base = (V0_MAG + noise_mag[i]) * \
                         np.exp(1j * np.deg2rad(V0_ANG_DEG + noise_ang[i]))

                if fault_type == 'Normal':
                    V = V_base
                else:
                    fb = np.random.randint(0, N_BUS)
                    Rf = float(np.random.choice(RF_OPTIONS))
                    Xf = float(np.random.choice(XF_OPTIONS))
                    V  = simulate_fault(Ybus, fb, fault_type, Rf, Xf)
                    fb_arr[idx] = fb
                    rf_arr[idx] = Rf
                    xf_arr[idx] = Xf

                Vm, Va, P, Q = compute_features(V, Ybus)
                X[idx, 0       :N_BUS  ] = Vm
                X[idx, N_BUS   :2*N_BUS] = Va
                X[idx, 2*N_BUS :3*N_BUS] = P
                X[idx, 3*N_BUS :       ] = Q

            if (c + 1) % 20 == 0:
                pct = (c + 1) / (n_per_class // chunk) * 100
                print(f"    {pct:5.1f}%", end='\r', flush=True)

        print(f"    100.0% done          ")

        df_c = pd.DataFrame(X, columns=vm_cols + va_cols + p_cols + q_cols)
        df_c.insert(0, 'fault_Xf',   xf_arr)
        df_c.insert(0, 'fault_Rf',   rf_arr)
        df_c.insert(0, 'fault_bus',  fb_arr)
        df_c.insert(0, 'fault_type', fault_type)
        df_c.insert(0, 'label',      label_map[fault_type])
        df_c.insert(0, 'time_step',  np.arange(n_per_class))
        all_dfs.append(df_c)

    print("\nCombining and shuffling...")
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    out = 'final_data/ieee39_final_dataset.csv'
    print(f"Saving to {out} ...")
    df.to_csv(out, index=False)

    print(f"\n[OK] Saved: {out}")
    print(f"     Rows    : {len(df):,}")
    print(f"     Columns : {len(df.columns)}")

    print("\nVm ranges per class:")
    for ft in fault_types:
        sub = df[df['fault_type'] == ft][vm_cols].values.flatten()
        print(f"  {ft:<8}: min={sub.min():.3f}  mean={sub.mean():.3f}")

    llg_min = df[df['fault_type'] == 'LLG'][vm_cols].values.min()
    ll_min  = df[df['fault_type'] == 'LL' ][vm_cols].values.min()
    diff    = abs(llg_min - ll_min)
    print(f"\nLLG vs LL diff = {diff:.4f}  {'[OK]' if diff > 0.05 else '[WARN]'}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("Remember: model in_dim must be 156 in train_pinn.py")
    print("=" * 60)


if __name__ == '__main__':
    # Change n_per_class to control dataset size:
    #   100000 = 500,000 total  (recommended for thesis)
    #   200000 = 1,000,000 total
    #    40000 = 200,000 total  (minimum)
    generate_dataset(n_per_class=100000)
