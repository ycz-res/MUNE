"""
Flask API 服务 - 提供 tasks.json 和 flow 状态的 REST API
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
from pathlib import Path
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # 允许跨域请求

TASKS_FILE = os.path.join(os.path.dirname(__file__), 'tasks.json')
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result')


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


@app.route('/api/results', methods=['GET'])
def list_results():
    """列出所有结果目录"""
    result_path = Path(RESULT_DIR)
    if not result_path.exists():
        return jsonify({'results': []})
    
    result_dirs = sorted([d for d in result_path.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
    
    results = []
    for dir_path in result_dirs:
        dir_info = {
            'name': dir_path.name,
            'modified': datetime.fromtimestamp(dir_path.stat().st_mtime).isoformat(),
            'has_train_visual': (dir_path / 'train_visual').exists(),
            'has_visual': (dir_path / 'visual').exists(),
            'has_train_json': (dir_path / f'train_{dir_path.name}.json').exists(),
            'has_test_json': (dir_path / f'test_{dir_path.name}.json').exists(),
        }
        
        # Count images
        train_visual_dir = dir_path / 'train_visual'
        if train_visual_dir.exists():
            dir_info['train_visual_count'] = len(list(train_visual_dir.glob('*.png')))
        
        visual_dir = dir_path / 'visual'
        if visual_dir.exists():
            dir_info['visual_count'] = len(list(visual_dir.glob('*.png')))
        
        results.append(dir_info)
    
    return jsonify({'results': results})


@app.route('/api/results/<timestamp>/train_visual', methods=['GET'])
def list_train_visuals(timestamp):
    """列出训练可视化图片"""
    result_dir = Path(RESULT_DIR) / timestamp
    train_visual_dir = result_dir / 'train_visual'
    
    if not train_visual_dir.exists():
        return jsonify({'images': []}), 404
    
    images = sorted([f.name for f in train_visual_dir.glob('*.png')])
    
    # Group by epoch
    images_by_epoch = {}
    for img_name in images:
        parts = img_name.replace('.png', '').split('_')
        if len(parts) >= 3 and parts[0] == 'train':
            try:
                epoch = int(parts[1])
                step = int(parts[2])
                if epoch not in images_by_epoch:
                    images_by_epoch[epoch] = []
                images_by_epoch[epoch].append({'name': img_name, 'step': step})
            except ValueError:
                continue
    
    return jsonify({
        'images': images,
        'by_epoch': {str(k): v for k, v in images_by_epoch.items()}
    })


@app.route('/api/results/<timestamp>/train_visual/<image_name>', methods=['GET'])
def get_train_visual(timestamp, image_name):
    """获取训练可视化图片"""
    result_dir = Path(RESULT_DIR) / timestamp
    image_path = result_dir / 'train_visual' / image_name
    
    if not image_path.exists() or not image_path.is_file():
        return jsonify({'error': 'image not found'}), 404
    
    return send_file(str(image_path), mimetype='image/png')


@app.route('/api/results/<timestamp>/visual', methods=['GET'])
def list_visuals(timestamp):
    """列出测试可视化图片"""
    result_dir = Path(RESULT_DIR) / timestamp
    visual_dir = result_dir / 'visual'
    
    if not visual_dir.exists():
        return jsonify({'images': []}), 404
    
    images = sorted([f.name for f in visual_dir.glob('*.png')])
    return jsonify({'images': images})


@app.route('/api/results/<timestamp>/visual/<image_name>', methods=['GET'])
def get_visual(timestamp, image_name):
    """获取测试可视化图片"""
    result_dir = Path(RESULT_DIR) / timestamp
    image_path = result_dir / 'visual' / image_name
    
    if not image_path.exists() or not image_path.is_file():
        return jsonify({'error': 'image not found'}), 404
    
    return send_file(str(image_path), mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

