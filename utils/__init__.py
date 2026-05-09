
import numpy as np


def build_Ybus(n_bus, branches, baseMVA=100):
    Ybus = np.zeros((n_bus, n_bus), dtype=np.complex128)
    for branch in branches:
        fbus = int(branch[0]) - 1
        tbus = int(branch[1]) - 1
        r = branch[2]
        x = branch[3]
        b = branch[4]
        ratio = branch[8] if branch[8] != 0 else 1.0
        Z = r + 1j * x
        Y = 1 / Z
        if ratio != 1.0:
            Y = Y / ratio
        Ybus[fbus, fbus] += Y + 1j * b / 2
        Ybus[tbus, tbus] += Y + 1j * b / 2
        Ybus[fbus, tbus] -= Y
        Ybus[tbus, fbus] -= Y
    return Ybus


def compute_power_injections(V, Ybus):
    S = V * np.conj(Ybus @ V)
    return np.real(S), np.imag(S)


def compute_voltage_residuals(V, Ybus, P_spec, Q_spec, v_indices, pq_indices):
    P_calc, Q_calc = compute_power_injections(V, Ybus)
    dP = P_spec[v_indices] - P_calc[v_indices]
    dQ = Q_spec[pq_indices] - Q_calc[pq_indices]
    return np.concatenate([dP, dQ])


def simulate_fault(V_normal, Ybus, fault_bus, fault_type, Rf=0.01, Xf=0.01):
    n_bus = len(V_normal)
    Zf = Rf + 1j * Xf
    Y_fault = np.zeros((n_bus, n_bus), dtype=np.complex128)
    
    if fault_type == '3LG':
        Y_fault[fault_bus, fault_bus] = 1 / Zf if Zf != 0 else 1e10
    elif fault_type == 'LG':
        Y_fault[fault_bus, fault_bus] = 1 / (3 * Zf) if Zf != 0 else 1e10
    elif fault_type == 'LLG':
        other_bus = (fault_bus + 1) % n_bus
        # 两相接地故障：每相对地都有阻抗，且两相之间也有连接
        Y_fault[fault_bus, fault_bus] = 2 / Zf if Zf != 0 else 2e10  # 增加接地路径
        Y_fault[other_bus, other_bus] = 2 / Zf if Zf != 0 else 2e10
        Y_fault[fault_bus, other_bus] = -1 / Zf if Zf != 0 else -1e10
        Y_fault[other_bus, fault_bus] = -1 / Zf if Zf != 0 else -1e10
    elif fault_type == 'LL':
        other_bus = (fault_bus + 1) % n_bus
        # 两相短路故障：只有相间短路，没有接地
        Y_fault[fault_bus, fault_bus] = 1 / Zf if Zf != 0 else 1e10
        Y_fault[other_bus, other_bus] = 1 / Zf if Zf != 0 else 1e10
        Y_fault[fault_bus, other_bus] = -1 / Zf if Zf != 0 else -1e10
        Y_fault[other_bus, fault_bus] = -1 / Zf if Zf != 0 else -1e10
    
    Y_total = Ybus + Y_fault
    try:
        V_fault = np.linalg.solve(Y_total, Ybus @ V_normal)
    except np.linalg.LinAlgError:
        V_fault = V_normal * 0.5
    
    return V_fault


def physics_loss_fn(Vm_hat, Va_hat, P_meas, Q_meas, Ybus, lambda_physics=0.1):
    batch_size = Vm_hat.shape[0]
    n_bus = Vm_hat.shape[1]
    V_hat = Vm_hat * torch.exp(1j * Va_hat)
    # 将 Ybus 转换为 PyTorch 张量，保持梯度流完整
    Ybus_tensor = torch.tensor(Ybus, dtype=torch.complex64, device=Vm_hat.device)
    P_hat, Q_hat = compute_power_injections_tensor(V_hat, Ybus_tensor)
    loss_p = torch.mean((P_hat - P_meas) ** 2)
    loss_q = torch.mean((Q_hat - Q_meas) ** 2)
    return lambda_physics * (loss_p + loss_q)


def compute_power_injections_tensor(V_tensor, Ybus_tensor):
    # 全程使用 PyTorch 张量操作，保持梯度流完整
    # V_tensor: (batch_size, n_bus) 类型的 complex 张量
    # Ybus_tensor: (n_bus, n_bus) 类型的 complex 张量
    I_tensor = torch.matmul(V_tensor, Ybus_tensor.t())
    S_tensor = V_tensor * torch.conj(I_tensor)
    P = torch.real(S_tensor).float()
    Q = torch.imag(S_tensor).float()
    return P, Q
