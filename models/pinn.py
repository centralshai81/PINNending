import numpy as np
import torch
import torch.nn as nn

class PINNBinary(nn.Module):
    def __init__(self, in_dim, n_bus, hidden=128, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.cls_head = nn.Linear(hidden, 2)
        self.state_head = nn.Linear(hidden, 2*n_bus)
        self.n_bus = n_bus

    def forward(self, x):
        h = self.backbone(x)
        logits = self.cls_head(h)
        state = self.state_head(h)
        Vm_hat = state[:, :self.n_bus]
        Va_hat = state[:, self.n_bus:]
        Vm_hat = 0.2 * torch.tanh(Vm_hat) + 1.0
        Va_hat = (10.0 * np.pi/180.0) * torch.tanh(Va_hat)
        return logits, Vm_hat, Va_hat


class PINNMultiClass(nn.Module):
    def __init__(self, in_dim, n_bus, n_classes=5, hidden=128, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.cls_head = nn.Linear(hidden, n_classes)
        self.state_head = nn.Linear(hidden, 2*n_bus)
        self.n_bus = n_bus
        self.n_classes = n_classes

    def forward(self, x):
        h = self.backbone(x)
        logits = self.cls_head(h)
        state = self.state_head(h)
        Vm_hat = state[:, :self.n_bus]
        Va_hat = state[:, self.n_bus:]
        Vm_hat = 0.2 * torch.tanh(Vm_hat) + 1.0
        Va_hat = (10.0 * np.pi/180.0) * torch.tanh(Va_hat)
        return logits, Vm_hat, Va_hat


class MLPBaseline(nn.Module):
    def __init__(self, in_dim, n_classes=5, hidden=128, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.cls_head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.cls_head(h)
        return logits