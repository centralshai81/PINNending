import numpy as np
import torch


def build_Ybus(n_bus, branches, baseMVA=100):
    """
    Build nodal admittance matrix from branch data.
    Handles transformer tap ratios and 1-indexed bus numbers.
    """
    Ybus = np.zeros((n_bus, n_bus), dtype=np.complex128)
    for branch in branches:
        fbus  = int(branch[0]) - 1    # 1-indexed -> 0-indexed
        tbus  = int(branch[1]) - 1
        r, x, b = branch[2], branch[3], branch[4]
        ratio = branch[8] if len(branch) > 8 and branch[8] != 0 else 1.0
        Z = r + 1j * x
        Y = (1.0 / Z) if abs(Z) > 1e-10 else (1e10 + 0j)
        Ybus[fbus, fbus] += Y / (ratio ** 2) + 1j * b / 2
        Ybus[tbus, tbus] += Y              + 1j * b / 2
        Ybus[fbus, tbus] -= Y / ratio
        Ybus[tbus, fbus] -= Y / ratio
    return Ybus


def compute_power_injections(V, Ybus):
    S = V * np.conj(Ybus @ V)
    return np.real(S), np.imag(S)


def simulate_fault(V_normal, Ybus, fault_bus, fault_type, Rf=0.01, Xf=0.01):
    """
    Simulate a short-circuit fault.
    LLG uses 2*Yf diagonal (distinct from LL which uses 1*Yf).
    """
    n_bus = len(V_normal)
    Zf    = Rf + 1j * Xf
    Yf    = (1.0 / Zf) if abs(Zf) > 1e-10 else 1e10
    Y_fault = np.zeros((n_bus, n_bus), dtype=np.complex128)

    if fault_type == '3LG':
        Y_fault[fault_bus, fault_bus] = Yf
    elif fault_type == 'LG':
        Y_fault[fault_bus, fault_bus] = Yf / 3.0
    elif fault_type == 'LLG':
        other = (fault_bus + 1) % n_bus
        Y_fault[fault_bus, fault_bus] += 2.0 * Yf   # ground paths
        Y_fault[other,     other    ] += 2.0 * Yf
        Y_fault[fault_bus, other    ]  = -Yf
        Y_fault[other,     fault_bus]  = -Yf
    elif fault_type == 'LL':
        other = (fault_bus + 1) % n_bus
        Y_fault[fault_bus, fault_bus] += 1.0 * Yf   # no ground path
        Y_fault[other,     other    ] += 1.0 * Yf
        Y_fault[fault_bus, other    ]  = -Yf
        Y_fault[other,     fault_bus]  = -Yf

    Y_total = Ybus + Y_fault
    try:
        V_fault = np.linalg.solve(Y_total, Ybus @ V_normal)
    except np.linalg.LinAlgError:
        V_fault = V_normal * 0.5
    return V_fault


def physics_loss_fn(Vm_hat, Va_hat, P_meas, Q_meas,
                    Ybus_input, lambda_physics=0.01):
    """
    Physics residual loss.

    Penalises the model when its predicted (Vm_hat, Va_hat) imply
    power injections that differ from the measured (P_meas, Q_meas).

    With voltage angles stored in the dataset, this residual is near
    zero for correct predictions, enabling effective PINN training.

    Feature layout expected by train_pinn.py:
        X[:, 0  :39 ] = vm
        X[:, 39 :78 ] = va    <- stored angles, NOT passed here
        X[:, 78 :117] = p     <- P_meas
        X[:, 117:156] = q     <- Q_meas
    """
    device = Vm_hat.device

    if not torch.is_tensor(Ybus_input):
        Ybus_tensor = torch.tensor(
            Ybus_input, dtype=torch.complex64, device=device
        )
    else:
        Ybus_tensor = Ybus_input.to(device)

    # Reconstruct complex voltage from predicted magnitudes and angles
    V_hat = Vm_hat.to(torch.complex64) * \
            torch.exp(1j * Va_hat.to(torch.complex64))

    P_hat, Q_hat = compute_power_injections_tensor(V_hat, Ybus_tensor)

    loss_p = torch.mean((P_hat - P_meas) ** 2)
    loss_q = torch.mean((Q_hat - Q_meas) ** 2)

    return lambda_physics * (loss_p + loss_q)


def compute_power_injections_tensor(V_tensor, Ybus_tensor):
    """
    Compute P and Q from complex voltages.
    Pure PyTorch — gradients flow correctly through this function.
    """
    I_tensor = torch.matmul(V_tensor, Ybus_tensor.t())
    S_tensor = V_tensor * torch.conj(I_tensor)
    P = torch.real(S_tensor).float()
    Q = torch.imag(S_tensor).float()
    return P, Q


def compute_voltage_residuals(V, Ybus, P_spec, Q_spec, v_indices, pq_indices):
    P_calc, Q_calc = compute_power_injections(V, Ybus)
    dP = P_spec[v_indices]  - P_calc[v_indices]
    dQ = Q_spec[pq_indices] - Q_calc[pq_indices]
    return np.concatenate([dP, dQ])
