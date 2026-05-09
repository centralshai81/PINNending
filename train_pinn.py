import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from models.pinn import PINNMultiClass

class FaultDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        self.fault_type_map = {'Normal': 0, '3LG': 1, 'LG': 2, 'LLG': 3, 'LL': 4}
        
        # 数据文件的列顺序是交错的：vm_0, p_0, q_0, vm_1, p_1, q_1...
        feature_cols = []
        for i in range(39):
            feature_cols.append(f'vm_{i}')
            feature_cols.append(f'p_{i}')
            feature_cols.append(f'q_{i}')
        
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df['fault_type'].map(self.fault_type_map).values
        
        print(f"数据加载完成: {len(self.X)} 个样本")
        print(f"故障类型分布: {np.bincount(self.y)}")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def train():
    print("="*60)
    print("开始训练 PINN 故障诊断模型")
    print("参数: hidden=128, depth=3")
    print("="*60)
    
    os.makedirs('outputs/checkpoints', exist_ok=True)
    
    dataset = FaultDataset('final_data/ieee39_final_dataset.csv')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = PINNMultiClass(in_dim=117, n_bus=39, n_classes=5, hidden=128, depth=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    n_epochs = 100
    best_acc = 0
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in dataloader:
            optimizer.zero_grad()
            logits, Vm_hat, Va_hat = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        
        acc = correct / total
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'outputs/checkpoints/best_model.pth')
            print(f"  ✓ 保存最佳模型 (准确率: {best_acc:.4f})")
    
    print("="*60)
    print(f"训练完成！最佳准确率: {best_acc:.4f}")
    print("="*60)

if __name__ == '__main__':
    train()