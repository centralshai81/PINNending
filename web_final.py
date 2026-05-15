import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.enhanced_agent import EnhancedDiagnosisAgent

app   = Flask(__name__)
agent = None


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
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
    print("\n=== Upload ===")
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400

        file = request.files['file']
        data = pd.read_csv(file)
        print(f"Rows: {len(data)}  Cols: {len(data.columns)}")

        # ── Detect feature format (156 with va, or 117 without va)
        has_va = 'va_0' in data.columns

        vm_cols = [f'vm_{i}' for i in range(39)]
        p_cols  = [f'p_{i}'  for i in range(39)]
        q_cols  = [f'q_{i}'  for i in range(39)]

        if has_va:
            va_cols        = [f'va_{i}' for i in range(39)]
            feature_cols   = vm_cols + va_cols + p_cols + q_cols  # 156
            print("[OK] 156-feature format detected (vm+va+p+q)")
        else:
            # Old 117-feature format — add zero angles
            va_cols      = None
            feature_cols = vm_cols + p_cols + q_cols               # 117
            print("[WARN] 117-feature format — va columns missing, using zeros")

        # Verify columns exist
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            return jsonify({'error': f'CSV missing columns: {missing[:5]}'}), 400

        samples = []
        for i in range(len(data)):
            samples.append({
                'index'         : i,
                'true_fault_type': str(data.loc[i, 'fault_type'])
                                   if 'fault_type' in data.columns else 'Unknown',
                'true_fault_bus': int(data.loc[i, 'fault_bus'])
                                  if 'fault_bus' in data.columns else -1,
            })

        if has_va:
            features = data[feature_cols].values.tolist()
        else:
            # Pad with zero voltage angles between vm and p
            vm_data = data[vm_cols].values
            va_data = np.zeros((len(data), 39), dtype=np.float32)
            p_data  = data[p_cols].values
            q_data  = data[q_cols].values
            features = np.concatenate(
                [vm_data, va_data, p_data, q_data], axis=1
            ).tolist()

        print(f"[OK] {len(samples)} samples, {len(features[0])} features each")

        return jsonify({
            'success'      : True,
            'total_samples': len(samples),
            'samples'      : samples,
            'features'     : features,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/diagnose', methods=['POST'])
def diagnose_v2():
    try:
        data         = request.get_json()
        features     = np.array(data['features'], dtype=np.float64)
        sample_index = int(data['sample_index'])

        result = get_agent().diagnose(features[sample_index])

        return jsonify(convert_numpy_types({
            'success'      : True,
            'result'       : result,
            'sample_index' : sample_index,
            'total_samples': len(features),
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/batch_analyze', methods=['POST'])
def batch_analyze():
    print("\n=== Batch Analyze ===")
    try:
        data     = request.get_json()
        features = np.array(data['features'], dtype=np.float64)
        print(f"Samples: {len(features)}")

        result = get_agent().batch_analyze(features)

        print("[OK] Batch analysis complete")
        return jsonify(convert_numpy_types({'success': True, 'result': result}))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/ask', methods=['POST'])
def ask_question():
    try:
        data     = request.get_json()
        question = data['question']
        context  = data.get('batch_result', None)

        answer = get_agent().answer_question(question, context)

        return jsonify(convert_numpy_types({'success': True, 'answer': answer}))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    get_agent()
    print("\n" + "=" * 60)
    print("Fault Diagnosis Agent — http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
