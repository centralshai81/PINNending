

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional
from models.pinn import PINNMultiClass

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.60   # below this -> uncertain
NORMAL_THRESHOLD     = 0.50   # Normal class must beat this to be called Normal


class EnhancedDiagnosisAgent:
    def __init__(self, data_path='final_data/ieee39_final_dataset.csv'):
        self.fault_type_cn = {
            'Normal'   : '正常运行',
            '3LG'      : '三相短路',
            'LG'       : '单相接地',
            'LLG'      : '两相接地',
            'LL'       : '两相短路',
            'Uncertain': '不确定',
        }
        self.idx_to_type = ['Normal', '3LG', 'LG', 'LLG', 'LL']
        self.fault_knowledge = {
            '3LG': '三相短路是最严重的短路故障，三相同时短路，电压跌落最大，故障电流也最大，通常由雷击、异物搭接引起。',
            'LG' : '单相接地是配电网最常见的故障，约占故障总数的70-80%，通常由绝缘子击穿、树木碰触引起。',
            'LLG': '两相接地是两相同时通过接地点形成回路，故障严重程度仅次于三相短路。',
            'LL' : '两相短路是两相间直接短路，没有接地，电压跌落程度介于LG和LLG之间。',
        }

        print("=" * 60)
        print("增强型电力系统故障诊断智能体")
        print("  in_dim=156 (vm+va+p+q)")
        print("=" * 60)

        self.n_bus = 39

        # Model expects 156 features (39 vm + 39 va + 39 p + 39 q)
        self.model = PINNMultiClass(
            in_dim=156, n_bus=39, n_classes=5, hidden=128, depth=3
        )
        self.model.load_state_dict(
            torch.load('outputs/checkpoints/best_model.pth', weights_only=True)
        )
        self.model.eval()
        print("[OK] PINN model loaded")

    def diagnose(self, features: np.ndarray) -> Dict:
        """
        Run inference on one 156-dimensional feature vector.

        Feature layout (must match training order):
            indices   0- 38 : vm_0...vm_38  (voltage magnitudes)
            indices  39- 77 : va_0...va_38  (voltage angles, radians)
            indices  78-116 : p_0...p_38    (active power, pu)
            indices 117-155 : q_0...q_38    (reactive power, pu)
        """
        if len(features) == 117:
            # Old 117-feature format detected — pad with zero angles
            # This allows backward compatibility with old CSV files
            vm = features[:39]
            p  = features[39:78]
            q  = features[78:117]
            va = np.zeros(39, dtype=np.float32)
            features = np.concatenate([vm, va, p, q]).astype(np.float32)

        vm_input = features[:39]
        min_vm   = float(np.min(vm_input))

        tensor_x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, Vm_hat, Va_hat = self.model(tensor_x)
            probs   = torch.softmax(logits, dim=1).numpy()[0]
            Vm_pred = Vm_hat.numpy()[0]

        pred_idx   = int(np.argmax(probs))
        top_prob   = float(probs[pred_idx])
        fault_type = self.idx_to_type[pred_idx]

        is_uncertain = top_prob < CONFIDENCE_THRESHOLD
        if is_uncertain and float(probs[0]) >= NORMAL_THRESHOLD:
            fault_type = 'Normal'
            pred_idx   = 0
            top_prob   = float(probs[0])

        has_fault = (fault_type != 'Normal') and \
                    (float(probs[0]) < NORMAL_THRESHOLD)

        class_probs = {
            cls: float(probs[i])
            for i, cls in enumerate(self.idx_to_type)
        }

        result: Dict = {
            'has_fault'          : has_fault,
            'fault_confidence'   : top_prob,
            'fault_type'         : fault_type,
            'fault_type_name'    : self.fault_type_cn.get(fault_type, fault_type),
            'class_probabilities': class_probs,
            'min_voltage'        : min_vm,
            'is_uncertain'       : is_uncertain,
        }

        if has_fault:
            voltage_deviation = np.abs(Vm_pred - 1.0)
            total_deviation   = float(np.sum(voltage_deviation))
            fault_bus         = int(np.argmax(voltage_deviation))
            loc_conf          = (
                float(voltage_deviation[fault_bus]) / total_deviation
                if total_deviation > 1e-6 else 0.0
            )
            result['fault_bus']               = fault_bus
            result['localization_confidence'] = loc_conf

        return result

    def batch_analyze(self, features_matrix: np.ndarray) -> Dict:
        n_samples = len(features_matrix)
        results   = []
        bus_fault_counts = {}

        for feat in features_matrix:
            res = self.diagnose(feat)
            results.append(res)
            if res['has_fault']:
                bus = res.get('fault_bus', -1)
                bus_fault_counts[bus] = bus_fault_counts.get(bus, 0) + 1

        fault_counts    = {t: 0 for t in self.idx_to_type}
        uncertain_count = 0
        for res in results:
            if res.get('is_uncertain'):
                uncertain_count += 1
            fault_counts[res['fault_type']] = \
                fault_counts.get(res['fault_type'], 0) + 1

        vm_array = np.array([f[:39].min() for f in features_matrix])

        summary = {
            'total_samples'  : n_samples,
            'fault_counts'   : fault_counts,
            'uncertain_count': uncertain_count,
            'fault_percentage': (n_samples - fault_counts.get('Normal', 0)) / n_samples,
            'top_fault_buses': sorted(
                bus_fault_counts.items(), key=lambda x: -x[1]
            )[:5],
            'results': results,
        }

        voltage_stats = {
            'mean': float(np.mean(vm_array)),
            'std' : float(np.std(vm_array)),
            'min' : float(np.min(vm_array)),
            'max' : float(np.max(vm_array)),
        }

        anomalies = [
            {'sample_index': i, 'reason': '低置信度诊断',
             'confidence': float(res['fault_confidence'])}
            for i, res in enumerate(results)
            if res['fault_confidence'] < CONFIDENCE_THRESHOLD
        ]

        fc = fault_counts
        n_fault = n_samples - fc.get('Normal', 0)
        report_lines = [
            f"共分析 {n_samples} 个样本",
            f"正常运行: {fc.get('Normal',0)} 个",
            f"故障样本: {n_fault} 个",
            f"  三相短路 (3LG): {fc.get('3LG',0)} 个",
            f"  单相接地  (LG): {fc.get('LG',0)} 个",
            f"  两相接地 (LLG): {fc.get('LLG',0)} 个",
            f"  两相短路  (LL): {fc.get('LL',0)} 个",
            f"平均电压: {voltage_stats['mean']:.3f} pu",
            f"最低电压: {voltage_stats['min']:.3f} pu",
        ]
        if uncertain_count > 0:
            report_lines.append(
                f"低置信度样本: {uncertain_count} 个 (置信度 < {CONFIDENCE_THRESHOLD*100:.0f}%)"
            )
        if summary['top_fault_buses']:
            b, c = summary['top_fault_buses'][0]
            report_lines.append(f"故障最多节点: #{b} 共 {c} 次")

        return {
            'summary'          : summary,
            'voltage_statistics': voltage_stats,
            'anomalies'        : anomalies,
            'report_lines'     : report_lines,
        }

    def answer_question(self, question: str,
                        batch_result: Optional[Dict] = None) -> Dict:
        q = question.lower().strip()

        if batch_result is None:
            if any(k in q for k in ['什么','介绍','解释','故障','短路','接地']):
                kb = self.answer_knowledge(q)
                if kb:
                    return kb
            return {'answer': '请先上传数据并完成批量分析，然后再提问。\n'
                              '也可以直接问电力系统故障知识，例如：什么是三相短路？'}

        s  = batch_result['summary']
        fc = s['fault_counts']
        vs = batch_result['voltage_statistics']
        n_fault   = s['total_samples'] - fc.get('Normal', 0)
        fault_rate = s['fault_percentage'] * 100

        answer = ""

        if self._match(q, ['总结','报告','概况','怎么样','情况']):
            answer = (
                f"共分析 {s['total_samples']} 个样本\n"
                f"正常: {fc.get('Normal',0)} 个 "
                f"({fc.get('Normal',0)/s['total_samples']*100:.1f}%)\n"
                f"故障: {n_fault} 个 ({fault_rate:.1f}%)\n\n"
                f"故障类型:\n"
                f"  3LG: {fc.get('3LG',0)}  LG: {fc.get('LG',0)}\n"
                f"  LLG: {fc.get('LLG',0)}  LL: {fc.get('LL',0)}\n\n"
                f"电压: 平均 {vs['mean']:.3f} pu  最低 {vs['min']:.3f} pu"
            )
        elif self._match(q, ['故障','多少','几个']):
            if self._match(q, ['比例','占比','率']):
                answer = f"共 {n_fault} 个故障样本，占比 {fault_rate:.1f}%"
            else:
                answer = f"共 {n_fault} 个故障样本"
        elif self._match(q, ['正常']):
            answer = f"正常样本 {fc.get('Normal',0)} 个，" \
                     f"占比 {fc.get('Normal',0)/s['total_samples']*100:.1f}%"
        elif self._match(q, ['三相','3lg']):
            answer = f"三相短路 (3LG): {fc.get('3LG',0)} 个"
        elif self._match(q, ['单相','lg']):
            answer = f"单相接地 (LG): {fc.get('LG',0)} 个"
        elif self._match(q, ['两相接地','llg']):
            answer = f"两相接地 (LLG): {fc.get('LLG',0)} 个"
        elif self._match(q, ['两相短路','ll']):
            answer = f"两相短路 (LL): {fc.get('LL',0)} 个"
        elif self._match(q, ['节点','位置','哪里','哪个']):
            if s['top_fault_buses']:
                b, c = s['top_fault_buses'][0]
                answer = f"故障最多的节点是 {b} 号节点，发生了 {c} 次"
            else:
                answer = "未检测到故障"
        elif self._match(q, ['电压','平均','最低','最高']):
            answer = (f"平均电压: {vs['mean']:.3f} pu\n"
                      f"最低电压: {vs['min']:.3f} pu\n"
                      f"最高电压: {vs['max']:.3f} pu")
        elif self._match(q, ['异常','置信度']):
            n = len(batch_result['anomalies'])
            answer = (f"发现 {n} 个低置信度样本，建议人工复核"
                      if n > 0 else "未发现低置信度异常样本")
        else:
            kb = self.answer_knowledge(q)
            if kb:
                return kb
            answer = ("暂时无法回答这个问题，请尝试：\n"
                      "• 总结分析结果\n• 故障有多少个？\n"
                      "• 什么是三相短路？")

        return {'answer': answer}

    def answer_knowledge(self, q: str) -> Optional[Dict]:
        if self._match(q, ['三相','3lg']):
            return {'answer': self.fault_knowledge['3LG'] +
                    "\n典型电压范围: 0.13~0.22 pu"}
        if self._match(q, ['单相','lg','接地']):
            return {'answer': self.fault_knowledge['LG'] +
                    "\n典型电压范围: 0.44~0.53 pu"}
        if self._match(q, ['两相接地','llg']):
            return {'answer': self.fault_knowledge['LLG'] +
                    "\n典型电压范围: 0.77~0.85 pu"}
        if self._match(q, ['两相短路','ll']):
            return {'answer': self.fault_knowledge['LL'] +
                    "\n典型电压范围: 0.68~0.76 pu"}
        return None

    @staticmethod
    def _match(q: str, keywords: List[str]) -> bool:
        return any(k in q for k in keywords)
