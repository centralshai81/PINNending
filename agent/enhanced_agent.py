import re
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple

from models.pinn import PINNMultiClass

# Base directory: project root (one level up from this file's agent/ folder)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIDENCE_THRESHOLD = 0.60
NORMAL_THRESHOLD     = 0.50


# ── Module-level helpers (outside the class) ──────────────────────────────────

_INTENT_PATTERNS: List[Tuple[str, List[str]]] = [
    ("summary", [
        r"总结|报告|概况|概览|整体|情况|怎么样|如何|overview|summary",
        r"分析结果|诊断结果|结果怎|结果如何",
    ]),
    ("fault_count", [
        r"故障.*(多少|几个|数量|总数|共有)",
        r"(多少|几个|数量|总数|共有).*故障",
        r"发生了多少",
    ]),
    ("fault_rate", [
        r"故障.*(比例|占比|率|百分)",
        r"(比例|占比|率|百分).*故障",
    ]),
    ("normal_count", [
        r"正常.*(多少|几个|数量|占比|比例|百分)",
        r"(多少|几个|数量).*(正常|无故障)",
    ]),
    ("type_3lg",  [r"三相|3lg|3LG|三相短路"]),
    ("type_lg",   [r"单相|单相接地|\bLG\b|\blg\b"]),
    ("type_llg",  [r"两相接地|LLG|llg"]),
    ("type_ll",   [r"两相短路|\bLL\b|\bll\b(?!g)"]),
    ("voltage_stats", [
        r"电压.*(平均|均值|最低|最高|统计|范围|情况)",
        r"(平均|均值|最低|最高).*(电压|pu)",
    ]),
    ("uncertainty", [
        r"低置信度|不确定|uncertain|置信度.*低|anomal",
        r"可信度|可靠性",
    ]),
    ("worst_fault", [
        r"最严重|最危险|最大.*故障|worst|most.*severe",
        r"最多.*故障类型|哪种故障.*多",
    ]),
    ("advice", [
        r"建议|怎么办|如何处理|措施|应对|recommend|suggest",
        r"需要.*注意|该.*怎么",
    ]),
    ("know_3lg",  [r"什么是.*(三相|3lg)|三相短路.*介绍|解释.*三相"]),
    ("know_lg",   [r"什么是.*(单相|单相接地|\bLG\b)|单相接地.*介绍"]),
    ("know_llg",  [r"什么是.*(两相接地|LLG)|两相接地.*介绍"]),
    ("know_ll",   [r"什么是.*(两相短路|\bLL\b)|两相短路.*介绍"]),
    ("know_pinn", [r"pinn|物理信息|神经网络|模型.*原理|怎么.*工作"]),
]

_KNOWLEDGE: Dict[str, str] = {
    "know_3lg": (
        "三相短路（3LG）是电力系统中最严重的短路故障。\n"
        "• 三相同时发生金属性短路，对称性最强。\n"
        "• 故障电流最大，电压跌落最剧烈，典型故障点电压 0.13~0.22 pu。\n"
        "• 常见原因：雷击、异物搭接、绝缘击穿。\n"
        "• 对系统稳定性威胁最大，继电保护通常以此为整定依据。"
    ),
    "know_lg": (
        "单相接地（LG）是配电网中最常见的故障类型，约占全部故障的 70~80%。\n"
        "• 一相经低阻抗（或金属性）接地，其余两相基本正常。\n"
        "• 故障电流相对较小，典型故障点电压 0.44~0.53 pu。\n"
        "• 常见原因：绝缘子击穿、树枝碰触、鸟害。\n"
        "• 中性点接地方式对故障电流大小有显著影响。"
    ),
    "know_llg": (
        "两相接地（LLG）是两相同时经接地点形成短路回路的故障。\n"
        "• 严重程度仅次于三相短路。\n"
        "• 典型故障点电压 0.77~0.85 pu，故障电流较大。\n"
        "• 属于不对称故障，需用对称分量法分析。\n"
        "• 常见原因：雷击引起两相绝缘同时击穿、导线脱落接地。"
    ),
    "know_ll": (
        "两相短路（LL）是两相之间直接短路、无接地的故障。\n"
        "• 不对称故障，故障电流约为三相短路的 √3/2 倍。\n"
        "• 典型故障点电压 0.68~0.76 pu。\n"
        "• 常见原因：导线弧垂过大相间放电、施工误碰。\n"
        "• 因无零序分量，零序保护不动作，需相间保护切除。"
    ),
    "know_pinn": (
        "本系统使用物理信息神经网络（PINN, Physics-Informed Neural Network）。\n"
        "• 输入：39 个节点的电压幅值（vm）、相角（va）、有功功率（p）、无功功率（q），共 156 维特征。\n"
        "• 输出：5 类分类（Normal / 3LG / LG / LLG / LL）+ 预测电压状态（Vm_hat, Va_hat）。\n"
        "• 物理约束：训练时加入功率流方程残差项，使模型预测符合基尔霍夫定律。\n"
        "• 两阶段训练：第一阶段纯分类损失，第二阶段逐步引入物理损失项（λ warmup）。"
    ),
}


def _classify_intent(question: str) -> str:
    """Return the best-matching intent label, or 'unknown'."""
    q = question.strip()
    for intent, patterns in _INTENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, q, re.IGNORECASE):
                return intent
    return "unknown"


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{n / total * 100:.1f}%"


def _fault_type_summary(fc: dict, total: int) -> str:
    mapping = [("3LG", "三相短路"), ("LG", "单相接地"),
               ("LLG", "两相接地"), ("LL", "两相短路")]
    lines = []
    for key, name in mapping:
        cnt = fc.get(key, 0)
        if cnt:
            lines.append(f"  · {name}（{key}）: {cnt} 个（{_pct(cnt, total)}）")
    return "\n".join(lines) if lines else "  · 无故障样本"


def _top_buses_text(top_buses: list) -> str:
    if not top_buses:
        return "未检测到故障节点。"
    parts = [f"#{b} 节点（{c} 次）" for b, c in top_buses[:5]]
    return "故障频发节点依次为：" + "、".join(parts) + "。"


def _dominant_fault(fc: dict) -> str:
    name_map = {"3LG": "三相短路", "LG": "单相接地",
                "LLG": "两相接地", "LL": "两相短路"}
    candidates = {k: v for k, v in fc.items() if k in name_map and v > 0}
    if not candidates:
        return ""
    top_key = max(candidates, key=candidates.get)
    return f"{name_map[top_key]}（{top_key}，共 {candidates[top_key]} 个）"


# ── Agent class ───────────────────────────────────────────────────────────────

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

        self.model = PINNMultiClass(
            in_dim=156, n_bus=39, n_classes=5, hidden=128, depth=3
        )
        self.model.load_state_dict(
            torch.load(os.path.join(_BASE_DIR, 'outputs', 'checkpoints', 'best_model.pth'), weights_only=True)
        )
        self.model.eval()
        print("[OK] PINN model loaded")

    def diagnose(self, features: np.ndarray) -> Dict:
        if len(features) == 117:
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
            'summary'           : summary,
            'voltage_statistics': voltage_stats,
            'anomalies'         : anomalies,
            'report_lines'      : report_lines,
        }

    def answer_question(self, question: str,
                        batch_result: Optional[Dict] = None) -> Dict:
        """Local intent-based Q&A — no external API calls."""
        intent = _classify_intent(question)

        # Knowledge questions never need batch data
        if intent.startswith("know_"):
            return {"answer": _KNOWLEDGE.get(intent, "暂无该知识条目。")}

        # Data-dependent questions with no data loaded
        if batch_result is None:
            if intent == "unknown":
                return {
                    "answer": (
                        "请先上传 CSV 数据并完成批量分析，然后再询问数据相关问题。\n"
                        "您也可以直接问故障知识，例如：\n"
                        "  • 什么是三相短路？\n"
                        "  • 什么是单相接地？\n"
                        "  • 什么是 PINN？"
                    )
                }
            return {
                "answer": (
                    f"您问的是\"{question}\"，但当前尚未上传数据。\n"
                    "请先上传 CSV 并点击「开始批量分析」，完成后即可回答。"
                )
            }

        s        = batch_result["summary"]
        fc       = s["fault_counts"]
        vs       = batch_result["voltage_statistics"]
        n        = s["total_samples"]
        n_normal = fc.get("Normal", 0)
        n_fault  = n - n_normal
        top_buses   = []  # fault bus display disabled
        uncertain   = s.get("uncertain_count", 0)
        anomalies   = batch_result.get("anomalies", [])

        if intent == "summary":
            dominant = _dominant_fault(fc)
            answer = (
                f"共分析 {n} 个样本：\n"
                f"  · 正常运行: {n_normal} 个（{_pct(n_normal, n)}）\n"
                f"  · 故障合计: {n_fault} 个（{_pct(n_fault, n)}）\n"
                f"\n故障类型分布：\n{_fault_type_summary(fc, n)}\n"
                f"\n电压统计：均值 {vs['mean']:.3f} pu，"
                f"最低 {vs['min']:.3f} pu，最高 {vs['max']:.3f} pu"
            )
            if uncertain:
                answer += f"\n\n低置信度样本：{uncertain} 个，建议人工复核。"
            if dominant:
                answer += f"\n最主要故障类型：{dominant}。"
            return {"answer": answer}

        if intent == "fault_count":
            return {"answer": (
                f"共检测到 {n_fault} 个故障样本（占总数 {_pct(n_fault, n)}）。\n"
                f"{_fault_type_summary(fc, n)}"
            )}

        if intent == "fault_rate":
            return {"answer": f"故障率为 {_pct(n_fault, n)}（{n_fault}/{n} 个样本）。"}

        if intent == "normal_count":
            return {"answer": f"正常运行样本共 {n_normal} 个，占比 {_pct(n_normal, n)}。"}

        if intent == "type_3lg":
            cnt = fc.get("3LG", 0)
            return {"answer": f"三相短路（3LG）样本：{cnt} 个（{_pct(cnt, n)}）。\n{_KNOWLEDGE['know_3lg'].splitlines()[0]}"}

        if intent == "type_lg":
            cnt = fc.get("LG", 0)
            return {"answer": f"单相接地（LG）样本：{cnt} 个（{_pct(cnt, n)}）。\n{_KNOWLEDGE['know_lg'].splitlines()[0]}"}

        if intent == "type_llg":
            cnt = fc.get("LLG", 0)
            return {"answer": f"两相接地（LLG）样本：{cnt} 个（{_pct(cnt, n)}）。\n{_KNOWLEDGE['know_llg'].splitlines()[0]}"}

        if intent == "type_ll":
            cnt = fc.get("LL", 0)
            return {"answer": f"两相短路（LL）样本：{cnt} 个（{_pct(cnt, n)}）。\n{_KNOWLEDGE['know_ll'].splitlines()[0]}"}

        if intent == "voltage_stats":
            return {"answer": (
                f"电压统计（全部样本）：\n"
                f"  · 均值:   {vs['mean']:.3f} pu\n"
                f"  · 最低:   {vs['min']:.3f} pu\n"
                f"  · 最高:   {vs['max']:.3f} pu\n"
                f"  · 标准差: {vs.get('std', 0):.3f} pu"
            )}

        if intent == "uncertainty":
            if uncertain == 0 and not anomalies:
                return {"answer": "未发现低置信度样本，所有诊断结果可信度正常。"}
            return {"answer": (
                f"共有 {uncertain} 个低置信度样本（置信度 < 60%），建议人工复核。\n"
                f"异常样本数：{len(anomalies)} 个。"
            )}

        if intent == "worst_fault":
            dominant = _dominant_fault(fc)
            return {"answer": (
                f"物理严重程度最高的故障类型是三相短路（3LG），"
                f"当前数据中共 {fc.get('3LG', 0)} 个。\n"
                f"数量最多的故障类型是：{dominant or '无故障'}。"
            )}

        if intent == "advice":
            lines = ["基于当前诊断结果，建议："]
            if fc.get("3LG", 0) > 0:
                lines.append(f"  · 检测到 {fc['3LG']} 个三相短路，应立即检查相关线路保护定值。")
            if fc.get("LG", 0) > 0:
                lines.append(f"  · {fc['LG']} 个单相接地故障，检查绝缘子和线路走廊。")
            if uncertain > 0:
                lines.append(f"  · {uncertain} 个低置信度样本需人工复核，勿自动跳闸。")
            if len(lines) == 1:
                lines.append("  · 当前无故障，系统运行正常，建议定期巡检。")
            return {"answer": "\n".join(lines)}

        # Unknown intent
        return {"answer": (
            "抱歉，未能理解该问题。您可以尝试以下问题：\n"
            "  · 总结分析结果\n"
            "  · 故障样本有多少个？\n"
            "  · 三相短路有几个？\n"
            "  · 平均电压是多少？\n"
            "  · 有多少低置信度样本？\n"
            "  · 有什么处置建议？\n"
            "  · 什么是三相短路 / 单相接地 / PINN？"
        )}

    def answer_knowledge(self, q: str) -> Optional[Dict]:
        """Kept for backward compatibility."""
        intent = _classify_intent(q)
        if intent.startswith("know_"):
            return {"answer": _KNOWLEDGE[intent]}
        return None