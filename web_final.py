from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.enhanced_agent import EnhancedDiagnosisAgent

app = Flask(__name__)

agent = None


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def get_agent():
    global agent
    if agent is None:
        agent = EnhancedDiagnosisAgent()
    return agent


@app.route('/')
def index():
    return render_template('final.html')


@app.route('/api/v2/upload', methods=['POST'])
def upload_file_v2():
    print("\n" + "="*60)
    print("=== API V2 上传文件 ===")
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        print(f"文件名: {file.filename}")
        
        data = pd.read_csv(file)
        print(f"数据行数: {len(data)}")
        print(f"包含的故障类型: {data['fault_type'].unique()}")
        
        # 训练数据的列顺序是按类型分组的：vm_0, vm_1..., p_0, p_1..., q_0, q_1...
        # 但测试数据的列顺序是交错的：vm_0, p_0, q_0, vm_1, p_1, q_1...
        # 需要将测试数据转换为训练数据的列顺序
        vm_cols = [f'vm_{i}' for i in range(39)]
        p_cols = [f'p_{i}' for i in range(39)]
        q_cols = [f'q_{i}' for i in range(39)]
        train_feature_cols = vm_cols + p_cols + q_cols  # 训练数据的列顺序
        
        samples = []
        for i in range(len(data)):
            sample_info = {
                'index': i,
                'true_fault_type': str(data.loc[i, 'fault_type']) if 'fault_type' in data.columns else 'Unknown',
                'true_fault_bus': int(data.loc[i, 'fault_bus']) if 'fault_bus' in data.columns else -1
            }
            samples.append(sample_info)
        
        # 按训练数据的列顺序提取特征
        features = data[train_feature_cols].values.tolist()
        
        print("="*60)
        
        return jsonify({
            'success': True,
            'total_samples': len(samples),
            'samples': samples,
            'features': features
        })
        
    except Exception as e:
        print(f"Error in upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/diagnose', methods=['POST'])
def diagnose_v2():
    try:
        data = request.get_json()
        features = np.array(data['features'], dtype=np.float64)
        sample_index = int(data['sample_index'])
        
        current_agent = get_agent()
        result = current_agent.diagnose(features[sample_index])
        
        return jsonify(convert_numpy_types({
            'success': True,
            'result': result,
            'sample_index': sample_index,
            'total_samples': len(features)
        }))
        
    except Exception as e:
        print(f"Error in diagnose: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/batch_analyze', methods=['POST'])
def batch_analyze():
    print("\n=== 批量数据分析 ===")
    try:
        data = request.get_json()
        features = np.array(data['features'], dtype=np.float64)
        print(f"分析样本数: {len(features)}")
        
        current_agent = get_agent()
        result = current_agent.batch_analyze(features)
        
        print("✓ 批量分析完成")
        return jsonify(convert_numpy_types({
            'success': True,
            'result': result
        }))
        
    except Exception as e:
        print(f"Error in batch_analyze: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data['question']
        context = data.get('batch_result', None)
        
        current_agent = get_agent()
        answer = current_agent.answer_question(question, context)
        
        return jsonify(convert_numpy_types({
            'success': True,
            'answer': answer
        }))
        
    except Exception as e:
        print(f"Error in ask: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    get_agent()
    print("\n" + "="*60)
    print("=== 故障诊断智能体服务 ===")
    print("访问: http://localhost:5000")
    print("="*60)
    print("核心功能:")
    print("  ✓ 单样本故障诊断")
    print("  ✓ 批量数据分析报告")
    print("  ✓ 数据异常检测")
    print("  ✓ 自然语言智能问答")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)
