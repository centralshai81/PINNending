import numpy as np
import torch


def build_Ybus(n_bus, branches, baseMVA=100):
    """Build nodal admittance matrix. 1-indexed buses converted to 0-indexed."""
    Ybus = np.zeros((n_bus, n_bus), dtype=np.complex128)
    for branch in branches:
        fbus  = int(branch[0]) - 1
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
    LLG: diagonal = 2*Yf (two ground paths)
    LL:  diagonal = 1*Yf (no ground path)
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
        Y_fault[fault_bus, fault_bus] += 2.0 * Yf
        Y_fault[other,     other    ] += 2.0 * Yf
        Y_fault[fault_bus, other    ]  = -Yf
        Y_fault[other,     fault_bus]  = -Yf
    elif fault_type == 'LL':
        other = (fault_bus + 1) % n_bus
        Y_fault[fault_bus, fault_bus] += 1.0 * Yf
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
    device = Vm_hat.device
    if not torch.is_tensor(Ybus_input):
        Ybus_tensor = torch.tensor(
            Ybus_input, dtype=torch.complex64, device=device
        )
    else:
        Ybus_tensor = Ybus_input.to(device)

    V_hat = Vm_hat.to(torch.complex64) * \
            torch.exp(1j * Va_hat.to(torch.complex64))

    P_hat, Q_hat = compute_power_injections_tensor(V_hat, Ybus_tensor)

    loss_p = torch.mean((P_hat - P_meas) ** 2)
    loss_q = torch.mean((Q_hat - Q_meas) ** 2)
    return lambda_physics * (loss_p + loss_q)


def compute_power_injections_tensor(V_tensor, Ybus_tensor):
    I_tensor = torch.matmul(V_tensor, Ybus_tensor.t())
    S_tensor = V_tensor * torch.conj(I_tensor)
    return torch.real(S_tensor).float(), torch.imag(S_tensor).float()
