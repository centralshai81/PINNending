
import numpy as np
import pandas as pd
import torch
import os
from typing import Dict, Tuple, List
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pinn import PINNBinary
from utils.physics import build_Ybus, physics_loss_fn


class ImprovedFaultDiagnosisAgent:
    def __init__(self, model_path, scaler_path, physics_path, device='cpu'):
        self.device = torch.device(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.config = checkpoint['config']
        self.input_dim = checkpoint['input_dim']
        self.n_bus = checkpoint['n_bus']
        
        self.model = PINNBinary(
            in_dim=self.input_dim,
            n_bus=self.n_bus,
            hidden=self.config['model']['hidden_dim'],
            depth=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout_rate']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        phys_data = np.load(physics_path)
        self.IEEE39_BRANCH = phys_data['IEEE39_BRANCH']
        self.Ybus = build_Ybus(self.n_bus, self.IEEE39_BRANCH)
        
        self.fault_type_names = {
            'Normal': '正常',
            '3LG': '三相短路',
            'LG': '单相接地',
            'LLG': '两相接地',
            'LL': '两相短路'
        }
    
    def detect_fault(self, features: np.ndarray) -> Tuple[bool, float]:
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        with torch.no_grad():
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            logits, _, _ = self.model(features_tensor)
            prob = torch.sigmoid(logits).item()
        
        is_fault = prob > 0.5
        confidence = prob if is_fault else 1 - prob
        
        return is_fault, confidence
    
    def localize_fault(self, features: np.ndarray) -> Tuple[int, float]:
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        with torch.no_grad():
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            _, Vm_hat, Va_hat = self.model(features_tensor)
            
            Vm_hat = Vm_hat.cpu().numpy().flatten()
            Va_hat = Va_hat.cpu().numpy().flatten()
            
            voltage_deviation = np.abs(Vm_hat - 1.0)
            fault_bus = np.argmax(voltage_deviation)
            confidence = voltage_deviation[fault_bus] / np.sum(voltage_deviation)
        
        return fault_bus, confidence
    
    def classify_fault_type(self, features: np.ndarray) -> Tuple[str, float]:
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        with torch.no_grad():
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            logits, Vm_hat, Va_hat = self.model(features_tensor)
            prob = torch.sigmoid(logits).item()
            
            Vm_hat = Vm_hat.cpu().numpy().flatten()
            Va_hat = Va_hat.cpu().numpy().flatten()
            
            voltage_drop = np.min(Vm_hat)
            angle_shift = np.max(np.abs(Va_hat))
            
            if prob < 0.5:
                fault_type = 'Normal'
                confidence = 1 - prob
            else:
                if voltage_drop < 0.2:
                    fault_type = '3LG'
                elif voltage_drop < 0.35:
                    fault_type = 'LLG'
                elif voltage_drop < 0.4:
                    fault_type = 'LL'
                else:
                    fault_type = 'LG'
                confidence = prob
        
        return fault_type, confidence
    
    def diagnose(self, features: np.ndarray) -> Dict:
        is_fault, fault_confidence = self.detect_fault(features)
        
        if is_fault:
            fault_bus, localization_confidence = self.localize_fault(features)
            fault_type_str, type_confidence = self.classify_fault_type(features)
            
            result = {
                'has_fault': True,
                'fault_confidence': fault_confidence,
                'fault_bus': fault_bus,
                'localization_confidence': localization_confidence,
                'fault_type': fault_type_str,
                'fault_type_name': self.fault_type_names.get(fault_type_str, fault_type_str),
                'type_confidence': type_confidence
            }
        else:
            result = {
                'has_fault': False,
                'fault_confidence': fault_confidence
            }
        
        return result
    
    def batch_diagnose(self, features_batch: np.ndarray) -> List[Dict]:
        results = []
        for features in features_batch:
            results.append(self.diagnose(features))
        return results
    
    def print_diagnosis(self, result: Dict):
        print("\n" + "="*80)
        print("故障诊断结果")
        print("="*80)
        
        if result['has_fault']:
            print(f"[警告] 检测到故障！")
            print(f"  故障置信度: {result['fault_confidence']:.2%}")
            print(f"\n[位置] 故障位置: 母线 {result['fault_bus']}")
            print(f"  位置置信度: {result['localization_confidence']:.2%}")
            print(f"\n[类型] 故障类型: {result['fault_type_name']} ({result['fault_type']})")
            print(f"  类型置信度: {result['type_confidence']:.2%}")
        else:
            print(f"[正常] 系统运行正常")
            print(f"  正常置信度: {result['fault_confidence']:.2%}")
        
        print("="*80 + "\n")


def load_improved_agent(model_path='outputs/checkpoints/pinn_best.pth', 
                        scaler_path='outputs/checkpoints/scaler.pkl',
                        physics_path='improved_data/ieee39_physics.npz',
                        device='cpu'):
    agent = ImprovedFaultDiagnosisAgent(model_path, scaler_path, physics_path, device)
    return agent


if __name__ == '__main__':
    agent = load_improved_agent()
    
    df = pd.read_csv('improved_data/ieee39_time_series_data.csv')
    feature_cols = [c for c in df.columns if c.startswith(('vm_', 'p_', 'q_'))]
    
    normal_idx = df[df['label'] == 0].index[0]
    fault_idx = df[df['label'] == 1].index[0]
    
    normal_sample = df.loc[normal_idx, feature_cols].values
    fault_sample = df.loc[fault_idx, feature_cols].values
    
    print("正常样本诊断:")
    result_normal = agent.diagnose(normal_sample)
    agent.print_diagnosis(result_normal)
    
    print("\n故障样本诊断:")
    result_fault = agent.diagnose(fault_sample)
    agent.print_diagnosis(result_fault)
