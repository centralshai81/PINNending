import numpy as np
import pandas as pd
import torch
from typing import Dict, List
from models.pinn import PINNMultiClass


class EnhancedDiagnosisAgent:
    def __init__(self, data_path='final_data/ieee39_final_dataset.csv'):
        self.fault_type_cn = {
            'Normal': '正常运行',
            '3LG': '三相短路',
            'LG': '单相接地',
            'LLG': '两相接地',
            'LL': '两相短路'
        }
        
        self.idx_to_type = ['Normal', '3LG', 'LG', 'LLG', 'LL']
        
        self.fault_knowledge = {
            '3LG': '三相短路是最严重的短路故障，三相同时短路，电压跌落最大，故障电流也最大，通常由雷击、异物搭接引起。',
            'LG': '单相接地是配电网最常见的故障，约占故障总数的70-80%，通常由绝缘子击穿、树木碰触引起。',
            'LLG': '两相接地是两相同时通过接地点形成回路，故障严重程度仅次于三相短路。',
            'LL': '两相短路是两相间直接短路，没有接地，电压跌落程度介于LG和LLG之间。'
        }
        
        print("=" * 60)
        print("增强型电力系统故障诊断智能体")
        print("✓ PINN推理  ✓ 批量分析  ✓ 数据洞察  ✓ 智能问答")
        print("=" * 60)
        
        self.n_bus = 39
        
        self.model = PINNMultiClass(in_dim=117, n_bus=39, n_classes=5, hidden=128, depth=3)
        self.model.load_state_dict(torch.load('outputs/checkpoints/best_model.pth', weights_only=True))
        self.model.eval()
        
        print("✓ PINN 模型权重加载完成")
    
    def diagnose(self, features: np.ndarray) -> Dict:
        # 训练数据的列顺序是按类型分组的：vm_0, vm_1..., p_0, p_1..., q_0, q_1...
        # 电压特征是前39个元素
        vm = features[:39]
        min_vm = np.min(vm)
        
        with torch.no_grad():
            tensor_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits, Vm_hat, Va_hat = self.model(tensor_features)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        fault_type = self.idx_to_type[pred_idx]
        confidence = float(probs[pred_idx])
        
        has_fault = fault_type != 'Normal'
        
        class_probs = {}
        for i, cls in enumerate(self.idx_to_type):
            class_probs[cls] = float(probs[i])
        
        result = {
            'has_fault': has_fault,
            'fault_confidence': confidence,
            'fault_type': fault_type,
            'fault_type_name': self.fault_type_cn[fault_type],
            'class_probabilities': class_probs,
            'min_voltage': float(min_vm)
        }
        
        if has_fault:
            # 使用模型预测的电压进行故障定位，而不是原始特征
            if Vm_hat is not None and len(Vm_hat) > 0:
                pred_vm = Vm_hat[0].numpy()
                voltage_deviation = np.abs(pred_vm - 1.0)
            else:
                voltage_deviation = np.abs(vm - 1.0)
            fault_bus = int(np.argmax(voltage_deviation))
            localization_confidence = float(voltage_deviation[fault_bus] / np.sum(voltage_deviation))
            
            result['fault_bus'] = int(fault_bus)
            result['localization_confidence'] = float(localization_confidence)
        
        return result
    
    def batch_analyze(self, features_matrix: np.ndarray) -> Dict:
        n_samples = len(features_matrix)
        
        results = []
        bus_fault_counts = {}
        
        for feat in features_matrix:
            res = self.diagnose(feat)
            results.append(res)
            if res['has_fault']:
                bus = res.get('fault_bus', -1)
                bus_fault_counts[bus] = bus_fault_counts.get(bus, 0) + 1
        
        fault_counts = {'Normal': 0, '3LG': 0, 'LG': 0, 'LLG': 0, 'LL': 0}
        for res in results:
            fault_counts[res['fault_type']] += 1
        
        # 训练数据的列顺序是按类型分组的，电压特征是前39个元素
        vm_array = np.array([f[:39].min() for f in features_matrix])
        
        summary = {
            'total_samples': n_samples,
            'fault_counts': fault_counts,
            'fault_percentage': (n_samples - fault_counts['Normal']) / n_samples,
            'top_fault_buses': sorted(bus_fault_counts.items(), key=lambda x: -x[1])[:5],
            'results': results
        }
        
        voltage_stats = {
            'mean': float(np.mean(vm_array)),
            'std': float(np.std(vm_array)),
            'min': float(np.min(vm_array)),
            'max': float(np.max(vm_array))
        }
        
        anomalies = []
        for i, res in enumerate(results):
            if res['fault_confidence'] < 0.7:
                anomalies.append({
                    'sample_index': i,
                    'reason': '低置信度诊断',
                    'confidence': float(res['fault_confidence'])
                })
        
        fc = fault_counts
        report_lines = [
            f"📊 共分析 {n_samples} 个样本",
            f"✅ 正常运行: {fc['Normal']} 个",
            f"🔴 故障样本: {n_samples - fc['Normal']} 个",
            f"   • 三相短路 (3LG): {fc['3LG']} 个",
            f"   • 单相接地 (LG): {fc['LG']} 个",
            f"   • 两相接地 (LLG): {fc['LLG']} 个",
            f"   • 两相短路 (LL): {fc['LL']} 个",
            f"⚡ 平均电压: {voltage_stats['mean']:.3f} pu",
            f"📉 最低电压: {voltage_stats['min']:.3f} pu"
        ]
        
        if len(summary['top_fault_buses']) > 0:
            report_lines.append(f"📍 故障最多的节点: #{summary['top_fault_buses'][0][0]} 共 {summary['top_fault_buses'][0][1]} 次")
        
        if len(anomalies) > 0:
            report_lines.append(f"⚠️  异常样本: {len(anomalies)} 个")
        
        return {
            'summary': summary,
            'voltage_statistics': voltage_stats,
            'anomalies': anomalies,
            'report_lines': report_lines
        }
    
    def answer_question(self, question: str, batch_result: Dict = None) -> Dict:
        question_lower = question.lower().strip()
        
        if batch_result is None:
            if any(k in question_lower for k in ['什么', '介绍', '解释', '说明', '故障', '短路', '接地']):
                return self.answer_knowledge(question_lower)
            return {'answer': '💡 请先上传数据并完成批量分析，然后我就能为您解答数据相关问题了！\n\n也可以直接问我电力系统故障知识，比如：\n• 什么是三相短路？\n• 单相接地是什么意思？'}
        
        s = batch_result['summary']
        fc = s['fault_counts']
        vs = batch_result['voltage_statistics']
        n_fault = s['total_samples'] - fc['Normal']
        fault_rate = s['fault_percentage'] * 100
        
        answer = ""
        
        if self._match(question_lower, ['你好', 'hello', 'hi', '嗨', '您好']):
            answer = "您好！我是电力系统故障诊断智能体 🤖\n\n我可以帮您：\n• 分析批量数据中的故障\n• 回答各种统计问题\n• 解释电力系统故障知识\n\n请问有什么可以帮助您的？"
        
        elif self._match(question_lower, ['多少', '几个', '总数', '样本']):
            if self._match(question_lower, ['故障']):
                if self._match(question_lower, ['比例', '占比', '百分之', '率']):
                    answer = f"🔴 在 {s['total_samples']} 个样本中，共有 {n_fault} 个故障样本，占比 {fault_rate:.1f}%。"
                else:
                    answer = f"🔴 共有 {n_fault} 个故障样本。"
            elif self._match(question_lower, ['正常']):
                answer = f"✅ 正常运行的样本有 {fc['Normal']} 个，占比 {fc['Normal']/s['total_samples']*100:.1f}%。"
            else:
                answer = f"📊 一共分析了 {s['total_samples']} 个样本。"
        
        elif self._match(question_lower, ['三相', '3lg']):
            cnt = fc['3LG']
            pct = cnt / s['total_samples'] * 100
            answer = f"⚡ 三相短路 (3LG) 有 {cnt} 个，占比 {pct:.1f}%。"
            if self._match(question_lower, ['介绍', '什么', '解释', '说明']):
                answer += "\n\n" + self.fault_knowledge['3LG']
        
        elif self._match(question_lower, ['单相', 'lg', '接地']):
            cnt = fc['LG']
            pct = cnt / s['total_samples'] * 100
            answer = f"⚡ 单相接地 (LG) 有 {cnt} 个，占比 {pct:.1f}%。"
            if self._match(question_lower, ['介绍', '什么', '解释', '说明']):
                answer += "\n\n" + self.fault_knowledge['LG']
        
        elif self._match(question_lower, ['两相接地', 'llg']):
            cnt = fc['LLG']
            pct = cnt / s['total_samples'] * 100
            answer = f"⚡ 两相接地 (LLG) 有 {cnt} 个，占比 {pct:.1f}%。"
            if self._match(question_lower, ['介绍', '什么', '解释', '说明']):
                answer += "\n\n" + self.fault_knowledge['LLG']
        
        elif self._match(question_lower, ['区别', '不同', '对比', '比较']):
            answer = """💡 四种短路故障的主要区别：

【电压跌落程度】（故障严重度）
3LG (0.13~0.22pu) > LLG (0.77~0.85pu) > LL (0.68~0.76pu) > LG (0.44~0.53pu)

【发生概率】
单相接地(LG) > 两相短路(LL) > 两相接地(LLG) > 三相短路(3LG)

【故障电流】
3LG > LLG > LL > LG

【诊断依据】
根据故障节点的最低电压幅值分类。"""
        
        elif self._match(question_lower, ['两相短路', 'll']):
            cnt = fc['LL']
            pct = cnt / s['total_samples'] * 100
            answer = f"⚡ 两相短路 (LL) 有 {cnt} 个，占比 {pct:.1f}%。"
            if self._match(question_lower, ['介绍', '什么', '解释', '说明']):
                answer += "\n\n" + self.fault_knowledge['LL']
        
        elif self._match(question_lower, ['最多', '最少', '哪个', '节点', '位置', '哪里']):
            if self._match(question_lower, ['故障', '发生', '出现']):
                if len(s['top_fault_buses']) > 0:
                    bus, cnt = s['top_fault_buses'][0]
                    answer = f"📍 故障最多的节点是 {bus} 号节点，共发生了 {cnt} 次故障。"
                else:
                    answer = "✅ 本次分析中没有检测到故障。"
        
        elif self._match(question_lower, ['电压', '平均', '最低', '最高', '幅值']):
            if self._match(question_lower, ['最低', '最小']):
                answer = f"📉 全局最低电压为 {vs['min']:.3f} pu。"
            elif self._match(question_lower, ['最高', '最大']):
                answer = f"📈 全局最高电压为 {vs['max']:.3f} pu。"
            else:
                answer = f"⚡ 平均电压为 {vs['mean']:.3f} pu，标准差 {vs['std']:.3f} pu。"
        
        elif self._match(question_lower, ['异常', '奇怪', '问题', '置信度']):
            n_anomaly = len(batch_result['anomalies'])
            if n_anomaly > 0:
                answer = f"⚠️  共发现 {n_anomaly} 个低置信度异常样本，建议人工复核。"
            else:
                answer = "✅ 本次分析未发现异常样本，所有诊断置信度都较高。"
        
        elif self._match(question_lower, ['总结', '报告', '概况', '怎么样', '如何', '情况']):
            answer = f"""📊 本次分析概况总结：

共分析 {s['total_samples']} 个样本：
  • 正常运行：{fc['Normal']} 个 ({fc['Normal']/s['total_samples']*100:.1f}%)
  • 故障样本：{n_fault} 个 ({fault_rate:.1f}%)

故障类型分布：
  • 三相短路 (3LG)：{fc['3LG']} 个
  • 单相接地 (LG)：{fc['LG']} 个
  • 两相接地 (LLG)：{fc['LLG']} 个
  • 两相短路 (LL)：{fc['LL']} 个

电压统计：
  • 平均电压：{vs['mean']:.3f} pu
  • 最低电压：{vs['min']:.3f} pu"""
        
        elif self._match(question_lower, ['最严重', '危害', '危险']):
            answer = """⚠️ 故障危害性排序：

1. 三相短路 (3LG) - 最严重，电压跌落最大，故障电流也最大
2. 两相接地 (LLG) - 两相同时故障，危害很大
3. 两相短路 (LL) - 相间直接短路
4. 单相接地 (LG) - 最常见，但相对危害较小

本次数据中三相短路有 {} 个。""".format(fc['3LG'])
        
        elif self._match(question_lower, ['最常见', '哪种多', '哪个多', '分布']):
            max_type = max(fc.items(), key=lambda x: x[1] if x[0] != 'Normal' else 0)
            answer = f"""📊 故障类型分布分析：

故障数量最多的是「{self.fault_type_cn[max_type[0]]}」，共 {max_type[1]} 个。

各种故障数量：
  • 单相接地 (LG)：{fc['LG']} 个
  • 两相短路 (LL)：{fc['LL']} 个
  • 两相接地 (LLG)：{fc['LLG']} 个
  • 三相短路 (3LG)：{fc['3LG']} 个"""
        
        elif self._match(question_lower, ['帮助', '可以', '能干', '会什么', '功能', '干啥', '能做', '你会']):
            answer = """🤖 我可以帮您解答以下问题：

【统计类】
• 一共有多少个样本？
• 故障有多少个？占比多少？
• 各种故障分别有多少个？

【故障节点】
• 故障最多的节点是哪个？

【电压统计】
• 平均电压是多少？
• 最低电压是多少？

【知识类】
• 什么是三相短路？
• 单相接地是什么？
• 哪种故障最严重？

【总结类】
• 给我一个总结报告
• 有什么异常情况吗？"""
        
        else:
            answer = self.answer_knowledge(question_lower)
            if answer is None:
                answer = """🤔 这个问题我暂时回答不了，您可以试试问：

• 总结一下分析结果
• 故障有多少个？
• 故障最多的节点是哪个？
• 平均电压是多少？
• 什么是三相短路？

或者点击上方的快捷问题按钮！"""
        
        return {'answer': answer}
    
    def answer_knowledge(self, q: str):
        if self._match(q, ['三相', '3lg']):
            return {'answer': "💡 " + self.fault_knowledge['3LG'] + "\n\n典型电压范围：0.126 ~ 0.215 pu"}
        
        if self._match(q, ['单相', 'lg', '接地']):
            return {'answer': "💡 " + self.fault_knowledge['LG'] + "\n\n典型电压范围：0.443 ~ 0.531 pu"}
        
        if self._match(q, ['两相接地', 'llg']):
            return {'answer': "💡 " + self.fault_knowledge['LLG'] + "\n\n典型电压范围：0.767 ~ 0.849 pu"}
        
        if self._match(q, ['两相短路', 'll']):
            return {'answer': "💡 " + self.fault_knowledge['LL'] + "\n\n典型电压范围：0.677 ~ 0.757 pu"}
        
        if self._match(q, ['什么是故障', '故障定义', '怎么诊断']):
            return {'answer': """💡 本系统的故障诊断原理：

基于物理信息神经网络(PINN)，利用电压跌落特征诊断：

1️⃣ 计算39个节点的电压幅值
2️⃣ 提取最低电压作为故障特征
3️⃣ 使用PINN分类器识别故障类型
4️⃣ 根据电压偏差最大的节点定位故障位置

五种状态：
• Normal (>0.96pu) - 正常运行
• 3LG (0.13~0.22pu) - 三相短路
• LG (0.44~0.53pu) - 单相接地
• LLG (0.77~0.85pu) - 两相接地
• LL (0.68~0.76pu) - 两相短路"""}
        
        return None
    
    def _match(self, q: str, keywords: List[str]) -> bool:
        return any(k in q for k in keywords)