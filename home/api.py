"""
Flask API 服务 - 提供 tasks.json 和 flow 状态的 REST API
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允许跨域请求

TASKS_FILE = 'tasks.json'
RESULT_DIR = 'result'


@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取所有任务"""
    if Path(TASKS_FILE).exists():
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'pending': [], 'running': [], 'completed': [], 'failed': [], 'cancelled': []})


@app.route('/api/tasks', methods=['POST'])
def update_tasks():
    """更新任务池"""
    data = request.json
    with open(TASKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return jsonify({'status': 'success'})


@app.route('/api/tasks/<task_name>/status', methods=['PUT'])
def update_task_status(task_name):
    """更新任务状态"""
    tasks = {}
    if Path(TASKS_FILE).exists():
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    
    new_status = request.json.get('status')
    if not new_status:
        return jsonify({'error': 'status required'}), 400
    
    # 找到任务并移动
    for status in ['pending', 'running', 'completed', 'failed', 'cancelled']:
        task = next((t for t in tasks.get(status, []) if t.get('name') == task_name), None)
        if task:
            tasks[status].remove(task)
            tasks.setdefault(new_status, []).append(task)
            task['status'] = new_status
            break
    
    with open(TASKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    return jsonify({'status': 'success'})


@app.route('/api/results/<timestamp>', methods=['GET'])
def get_result(timestamp):
    """获取实验结果"""
    result_dir = Path(RESULT_DIR) / timestamp
    if not result_dir.exists():
        return jsonify({'error': 'result not found'}), 404
    
    result = {}
    
    # 读取训练结果
    train_json = result_dir / f'train_{timestamp}.json'
    if train_json.exists():
        with open(train_json, 'r') as f:
            result['train'] = json.load(f)
    
    # 读取测试结果
    test_json = result_dir / f'test_{timestamp}.json'
    if test_json.exists():
        with open(test_json, 'r') as f:
            result['test'] = json.load(f)
    
    return jsonify(result)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取统计信息"""
    tasks = {}
    if Path(TASKS_FILE).exists():
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    
    return jsonify({
        'pending': len(tasks.get('pending', [])),
        'running': len(tasks.get('running', [])),
        'completed': len(tasks.get('completed', [])),
        'failed': len(tasks.get('failed', [])),
        'cancelled': len(tasks.get('cancelled', [])),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

