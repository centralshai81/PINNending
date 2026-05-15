import sys, io
import torch
import numpy as np
import pandas as pd
from models.pinn import PINNMultiClass

# Ensure utf-8 encoding for the terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=========================================")
print("  PINN Model Ultimate Diagnostic Script  ")
print("=========================================\n")

# ==========================================
# Part 1: Auto-detect the training dimension
# ==========================================
print("▶ PART 1: Detecting model training dimensions...")
valid_dim = None

for dim in [117, 156]:
    try:
        m = PINNMultiClass(in_dim=dim, n_bus=39, n_classes=5, hidden=128, depth=3)
        m.load_state_dict(torch.load('outputs/checkpoints/best_model.pth', weights_only=True))
        print(f"  [OK] in_dim={dim} loads successfully! (Model was trained with {dim} features)")
        valid_dim = dim
    except Exception as e:
        print(f"  [X]  in_dim={dim} fails to load ({str(e)[:50]}...)")

# ==========================================
# Part 2: Output real prediction probabilities
# ==========================================
print("\n▶ PART 2: Verifying probability outputs on DEMO data...")

if valid_dim is None:
    print("  ❌ FATAL ERROR: Model failed to load. Please check if best_model.pth is corrupted.")
else:
    # Load model with the auto-detected valid dimension
    model = PINNMultiClass(in_dim=valid_dim, n_bus=39, n_classes=5, hidden=128, depth=3)
    model.load_state_dict(torch.load('outputs/checkpoints/best_model.pth', weights_only=True))
    model.eval()

    # Load demo data
    df = pd.read_csv('DEMO_PERFECT.csv')
    has_va = 'va_0' in df.columns

    vm = [f'vm_{i}' for i in range(39)]
    va = [f'va_{i}' for i in range(39)]
    p  = [f'p_{i}'  for i in range(39)]
    q  = [f'q_{i}'  for i in range(39)]

    # Auto-select columns based on the presence of phase angles (va)
    fc = vm + va + p + q if has_va else vm + p + q
    print(f"  Current data feature count: {len(fc)} | Contains phase angles (va): {has_va}\n")

    # Run inference row by row
    print(f"  {'True Label':<10} -> {'Predicted':<9} | {'Internal Probabilities [N, 3LG, LG, LLG, LL]'}")
    print("-" * 75)
    
    for i in range(len(df)):
        x = torch.tensor(df[fc].iloc[i].values, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            probs = torch.softmax(model(x)[0], dim=1).numpy()[0]
        
        true_label = df.iloc[i]["fault_type"]
        pred_label = ["N", "3LG", "LG", "LLG", "LL"][np.argmax(probs)]
        probs_rounded = [round(float(p), 3) for p in probs]
        
        print(f"  {true_label:<10} -> {pred_label:<9} | {probs_rounded}")

print("\n=========================================")
print("  Diagnosis Complete!")