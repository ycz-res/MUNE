"""
Task Pool Visualization Dashboard - Streamlit
Integrated with task execution engine
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import subprocess
import time
import sys
import os
import threading
import shutil
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASKS_FILE = os.path.join(os.path.dirname(__file__), 'tasks.json')
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result')


@st.cache_data(ttl=1)
def load_tasks():
    """Load task pool"""
    if Path(TASKS_FILE).exists():
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'pending': [], 'running': [], 'completed': [], 'failed': [], 'cancelled': []}


def save_tasks(tasks):
    """Save task pool"""
    with open(TASKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)


def move_task(tasks, name, from_status, to_status, **kwargs):
    """Move task between statuses"""
    task = next((t for t in tasks.get(from_status, []) if t.get('name') == name), None)
    if task:
        task.update(kwargs)
        tasks[from_status] = [t for t in tasks[from_status] if t.get('name') != name]
        tasks.setdefault(to_status, []).append(task)
        return True
    return False


def build_args(task, script_name, timestamp=None, result_dir=None):
    """Build command line arguments dynamically"""
    args = ['python3', script_name]
    
    # Fields to skip (metadata fields not used by training scripts)
    skip_fields = {
        'name', 'timestamp', 'started_at', 'completed_at', 
        'error', 'result_dir'  # result_dir is handled separately
    }
    
    # Valid training script arguments (from train.py argparse)
    valid_train_args = {
        'batch_size', 'epochs', 'num_workers', 'pin_memory', 'device',
        'lr', 'weight_decay', 'grad_clip', 'patience', 'loss_type',
        'model_type', 'save_best', 'threshold_mode', 'dataset_type',
        'metrics_threshold', 'use_weighted_loss', 'pos_weight', 'd_model',
        'lr_scheduler', 'warmup_epochs', 'dropout'
    }
    
    # Valid test script arguments (we'll filter based on script_name)
    valid_test_args = {
        'model_type', 'threshold_mode', 'dataset_type', 'metrics_threshold',
        'hidden_size', 'result_dir', 'timestamp'
    }
    
    # Choose which valid args to use based on script
    if script_name == 'train.py':
        valid_args = valid_train_args
    elif script_name == 'test.py':
        valid_args = valid_test_args
    else:
        valid_args = set()  # visualization.py doesn't need task args
    
    # Debug: log what we're processing
    logger.debug(f"Building args for {script_name} with task keys: {list(task.keys())}")
    
    for key, value in task.items():
        # Skip metadata fields first
        if key in skip_fields:
            logger.debug(f"Skipping metadata field: {key}")
            continue
        
        # Only include valid arguments for the script
        if script_name in ['train.py', 'test.py'] and key not in valid_args:
            logger.debug(f"Skipping invalid field for {script_name}: {key}")
            continue
        
        if isinstance(value, bool):
            args.extend([f'--{key}', str(value)])
        else:
            args.extend([f'--{key}', str(value)])
    
    if timestamp:
        args.extend(['--timestamp', timestamp])
    if result_dir:
        args.extend(['--result_dir', str(result_dir)])
    
    logger.debug(f"Final args: {args}")
    return args


def run_cmd(args, log_file=None):
    """Execute command"""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w') as f:
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                               text=True, bufsize=1, cwd=os.path.dirname(os.path.dirname(__file__)))
            try:
                # Read all output lines
                while True:
                    line = p.stdout.readline()
                    if not line and p.poll() is not None:
                        break
                    if line:
                        f.write(line.rstrip() + '\n')
                        f.flush()
            except Exception as e:
                logger.error(f"Error reading output: {e}")
            finally:
                # Ensure process is terminated
                if p.poll() is None:
                    p.terminate()
                    p.wait()
                
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, args)
    else:
        subprocess.run(args, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))


def execute_task(task, result_dir, status_container=None):
    """Execute a single task (train -> test -> visualization)"""
    name = task['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure result_dir is absolute path
    if not result_dir or not result_dir.strip():
        result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result')
    result_dir = os.path.abspath(result_dir)
    
    # Don't create directory until we actually start training
    task_log_dir = Path(result_dir) / timestamp
    
    # Create a clean copy of the task, removing any metadata fields
    clean_task = {k: v for k, v in task.items() 
                  if k not in ['started_at', 'completed_at', 'error', 'timestamp']}
    
    def update_status(message, status_type='info'):
        """Update status display"""
        if status_container:
            if status_type == 'error':
                status_container.error(f"❌ {message}")
            elif status_type == 'success':
                status_container.success(f"✅ {message}")
            else:
                status_container.info(f"ℹ️ {message}")
    
    try:
        update_status(f"Starting task: {name}")
        tasks = load_tasks()
        
        # Check if task is already running
        running_tasks = [t for t in tasks.get('running', []) if t.get('name') == name]
        if running_tasks:
            raise Exception(f"Task '{name}' is already running")
        
        # Save original task before moving (to avoid started_at contamination)
        original_task = next((t for t in tasks.get('pending', []) if t.get('name') == name), None)
        if not original_task:
            raise Exception(f"Task '{name}' not found in pending tasks")
        
        # Create a clean copy of original_task too
        original_task = {k: v for k, v in original_task.items() 
                         if k not in ['started_at', 'completed_at', 'error', 'timestamp']}
        
        # Move to running BEFORE creating directory
        move_task(tasks, name, 'pending', 'running', 
                 started_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        save_tasks(tasks)
        
        # Training
        update_status("Starting training phase...")
        # Create directory only when training actually starts (after moving to running)
        task_log_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        # Filter out metadata fields before building args - use original_task
        train_task = {k: v for k, v in original_task.items() 
                     if k not in ['name']}
        try:
            logger.info(f"Running command: {' '.join(build_args(train_task, 'train.py', timestamp, original_task.get('result_dir', 'result')))}")
            run_cmd(build_args(train_task, 'train.py', timestamp, original_task.get('result_dir', 'result')), 
                   task_log_dir / 'train.log')
            train_time = (time.time() - start) / 60
            update_status(f"Training completed in {train_time:.2f} minutes", 'success')
        except subprocess.CalledProcessError as e:
            error_msg = f"Training failed with exit code {e.returncode}"
            update_status(f"Training failed: {error_msg}", 'error')
            log_file = task_log_dir / 'train.log'
            if log_file.exists() and log_file.stat().st_size > 0:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        update_status(f"Last log lines:\n{''.join(lines[-10:])}", 'error')
            else:
                update_status(f"Log file is empty - command may not have started", 'error')
            raise Exception(f"{error_msg}\nCheck log: {log_file}")
        except Exception as e:
            update_status(f"Training failed: {str(e)}", 'error')
            log_file = task_log_dir / 'train.log'
            if log_file.exists() and log_file.stat().st_size > 0:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        update_status(f"Last log lines:\n{''.join(lines[-10:])}", 'error')
            raise
        
        # Testing
        update_status("Starting testing phase...")
        start = time.time()
        # Filter out training-specific fields and metadata - use original_task
        test_task = {k: v for k, v in original_task.items() 
                    if k not in ['name', 'batch_size', 'epochs', 'lr', 'patience', 'warmup_epochs', 
                               'dropout', 'lr_scheduler', 'grad_clip', 'use_weighted_loss', 
                               'pos_weight', 'started_at', 'completed_at', 'error', 'timestamp']}
        if 'd_model' in test_task:
            test_task['hidden_size'] = test_task.pop('d_model')
        try:
            run_cmd(build_args(test_task, 'test.py', timestamp, result_dir),
                   task_log_dir / 'test.log')
            test_time = time.time() - start
            update_status(f"Testing completed in {test_time:.2f} seconds", 'success')
        except Exception as e:
            update_status(f"Testing failed: {str(e)}", 'error')
            log_file = task_log_dir / 'test.log'
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        update_status(f"Last log lines:\n{''.join(lines[-10:])}", 'error')
            raise
        
        # Visualization
        update_status("Starting visualization phase...")
        start = time.time()
        try:
            run_cmd(build_args({}, 'visualization.py', timestamp, result_dir),
                   task_log_dir / 'visualization.log')
            viz_time = time.time() - start
            update_status(f"Visualization completed in {viz_time:.2f} seconds", 'success')
        except Exception as e:
            update_status(f"Visualization failed: {str(e)}", 'error')
            log_file = task_log_dir / 'visualization.log'
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        update_status(f"Last log lines:\n{''.join(lines[-10:])}", 'error')
            raise
        
        update_status(f"Task '{name}' completed successfully!", 'success')
        tasks = load_tasks()
        move_task(tasks, name, 'running', 'completed',
                 completed_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 timestamp=timestamp)
        save_tasks(tasks)
        
    except Exception as e:
        error_msg = str(e)
        update_status(f"Task '{name}' failed: {error_msg}", 'error')
        tasks = load_tasks()
        move_task(tasks, name, 'running', 'failed',
                 completed_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 error=error_msg)
        save_tasks(tasks)
        raise


def cleanup_empty_dirs(result_dir):
    """Clean up empty result directories"""
    result_path = Path(result_dir)
    if not result_path.exists():
        return
    
    cleaned = 0
    for item in result_path.iterdir():
        if item.is_dir():
            # Check if directory is empty or only has empty log files
            files = list(item.iterdir())
            has_content = False
            for f in files:
                if f.is_file() and f.stat().st_size > 0:
                    has_content = True
                    break
            
            if not has_content:
                try:
                    shutil.rmtree(item)
                    cleaned += 1
                except Exception:
                    pass
    
    return cleaned


def run_flow_background():
    """Run flow in background thread"""
    max_retries = 3
    retry_count = 0
    
    while True:
        # Check if auto flow should continue
        if not st.session_state.get('auto_flow_running', False):
            logger.info("Auto flow stopped by user")
            break
        
        tasks = load_tasks()
        if not tasks.get('pending', []):
            logger.info("No pending tasks, stopping auto flow")
            st.session_state.auto_flow_running = False
            break
        
        # Check if there's already a running task
        if tasks.get('running', []):
            logger.info("Task already running, waiting...")
            time.sleep(5)  # Wait longer if task is running
            continue
        
        task = tasks['pending'][0]
        task_name = task.get('name', 'Unknown')
        logger.info(f"Starting task: {task_name}")
        
        try:
            execute_task(task, RESULT_DIR)
            # Reset retry count on success
            retry_count = 0
        except Exception as e:
            logger.error(f"Task failed: {e}")
            retry_count += 1
            
            # If task failed multiple times, stop auto flow to prevent infinite loop
            if retry_count >= max_retries:
                logger.error(f"Task failed {max_retries} times, stopping auto flow")
                st.session_state.auto_flow_running = False
                break
            
            # Task should be moved to failed by execute_task
            # Wait before retrying next task
            time.sleep(10)
        
        time.sleep(2)  # Wait between tasks


# Initialize session state
if 'execution_status' not in st.session_state:
    st.session_state.execution_status = {}
if 'auto_flow_running' not in st.session_state:
    st.session_state.auto_flow_running = False

# Page config
st.set_page_config(
    page_title="Task Pool Manager",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
    .task-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: #fafafa;
    }
    </style>
""", unsafe_allow_html=True)

# Header
col_header, col_refresh = st.columns([4, 1])
with col_header:
    st.markdown('<h1 class="main-header">🚀 Task Pool Manager</h1>', unsafe_allow_html=True)
with col_refresh:
    if st.button("🔄", help="Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Status bar
tasks = load_tasks()
running_count = len(tasks.get('running', []))
status_col1, status_col2 = st.columns([3, 1])

with status_col1:
    if st.session_state.auto_flow_running:
        st.success(f"🔄 Auto Flow Running • {running_count} Active Task(s)")
    elif running_count > 0:
        st.info(f"⚙️ {running_count} Task(s) Running")
    else:
        st.info("💤 No Active Tasks")

with status_col2:
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

st.divider()

# Statistics cards
st.subheader("📊 Overview")
metric_cols = st.columns(5)

stats_config = [
    ("⏳ Pending", len(tasks.get('pending', [])), "#FFA726"),
    ("🚀 Running", len(tasks.get('running', [])), "#42A5F5"),
    ("✅ Completed", len(tasks.get('completed', [])), "#66BB6A"),
    ("❌ Failed", len(tasks.get('failed', [])), "#EF5350"),
    ("🚫 Cancelled", len(tasks.get('cancelled', [])), "#78909C")
]

for idx, (label, value, color) in enumerate(stats_config):
    with metric_cols[idx]:
        st.metric(label, value, delta=None)

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⏳ Pending", "🚀 Running", "✅ Completed", "❌ Failed", "➕ New Task"
])

# Pending tasks
with tab1:
    pending = tasks.get('pending', [])
    if pending:
        for task in pending:
            task_name = task.get('name', 'Unknown')
            with st.container():
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f"### 📋 {task_name}")
                    with st.expander("View Details", expanded=False):
                        st.json(task, expanded=False)
                with cols[1]:
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        # Check if task is already running
                        is_running = any(t.get('name') == task_name for t in tasks.get('running', []))
                        if is_running:
                            st.button("⏸️ Running...", key=f"start_{task_name}", use_container_width=True, disabled=True)
                        else:
                            if st.button("▶️ Start", key=f"start_{task_name}", use_container_width=True, type="primary"):
                                # Double check task is still pending before starting
                                current_tasks = load_tasks()
                                if not any(t.get('name') == task_name for t in current_tasks.get('pending', [])):
                                    st.warning(f"Task '{task_name}' is no longer pending")
                                    st.rerun()
                                else:
                                    with st.status(f"🚀 Executing: {task_name}", expanded=True) as status:
                                        try:
                                            execute_task(task, RESULT_DIR, status)
                                            status.update(label=f"✅ Completed: {task_name}", state="complete")
                                        except Exception as e:
                                            status.update(label=f"❌ Failed: {task_name}", state="error")
                                            st.error(f"**Error:** {str(e)}")
                                            # Find the most recent timestamp directory
                                            result_path = Path(RESULT_DIR)
                                            if result_path.exists():
                                                recent_dirs = sorted(result_path.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)[:1]
                                                for log_dir in recent_dirs:
                                                    for log_file in ['train.log', 'test.log', 'visualization.log']:
                                                        log_path = log_dir / log_file
                                                        if log_path.exists() and log_path.stat().st_size > 0:
                                                            with open(log_path, 'r') as f:
                                                                lines = f.readlines()
                                                                if lines:
                                                                    st.text_area(f"Error log ({log_file}):", 
                                                                               ''.join(lines[-20:]), 
                                                                               height=150, 
                                                                               key=f"error_log_{task_name}_{log_file}_{log_dir.name}")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()
                    
                    with btn_col2:
                        if st.button("🚫 Cancel", key=f"cancel_{task_name}", use_container_width=True):
                            tasks = load_tasks()
                            if move_task(tasks, task_name, 'pending', 'cancelled'):
                                save_tasks(tasks)
                                st.cache_data.clear()
                                st.success("✅ Cancelled")
                                time.sleep(0.5)
                                st.rerun()
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_pending_{task_name}", use_container_width=True):
                        tasks = load_tasks()
                        tasks['pending'] = [t for t in tasks.get('pending', []) if t.get('name') != task_name]
                        save_tasks(tasks)
                        st.cache_data.clear()
                        st.success("✅ Deleted")
                        time.sleep(0.5)
                        st.rerun()
                st.divider()
    else:
        st.info("✨ No pending tasks")

# Running tasks
with tab2:
    running = tasks.get('running', [])
    if running:
        for task in running:
            task_name = task.get('name', 'Unknown')
            with st.container():
                st.markdown(f"### 🚀 {task_name}")
                col1, col2 = st.columns([3, 1])
                with col1:
                    with st.expander("View Details", expanded=False):
                        st.json(task, expanded=False)
                    if task.get('started_at'):
                        st.caption(f"⏰ Started: {task.get('started_at')}")
                with col2:
                    # Action buttons
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("⏹️ Stop", key=f"stop_{task_name}", use_container_width=True):
                            tasks = load_tasks()
                            if move_task(tasks, task_name, 'running', 'cancelled',
                                       stopped_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')):
                                save_tasks(tasks)
                                st.cache_data.clear()
                                st.success("✅ Task stopped")
                                time.sleep(0.5)
                                st.rerun()
                    
                    with btn_col2:
                        if st.button("🔁 Retry", key=f"retry_{task_name}", use_container_width=True):
                            tasks = load_tasks()
                            # Move to pending and reset status
                            task_copy = next((t for t in tasks.get('running', []) if t.get('name') == task_name), None)
                            if task_copy:
                                # Remove status fields
                                for key in ['started_at', 'completed_at', 'error', 'timestamp', 'stopped_at']:
                                    task_copy.pop(key, None)
                                
                                if move_task(tasks, task_name, 'running', 'pending'):
                                    save_tasks(tasks)
                                    st.cache_data.clear()
                                    st.success("✅ Task moved to pending")
                                    time.sleep(0.5)
                                    st.rerun()
                    
                    timestamp = task.get('timestamp')
                    if timestamp:
                        result_path = Path(RESULT_DIR) / timestamp
                        if result_path.exists():
                            st.success(f"📁 {timestamp}")
                            if st.button("📊 View Logs", key=f"view_log_{task_name}", use_container_width=True):
                                log_file = result_path / 'train.log'
                                if log_file.exists():
                                    with open(log_file, 'r') as f:
                                        st.text_area("Training Log", f.read(), height=300, key=f"log_{task_name}")
                st.divider()
    else:
        st.info("✨ No running tasks")

# Completed tasks
with tab3:
    completed = tasks.get('completed', [])
    if completed:
        completed_sorted = sorted(completed, key=lambda x: x.get('completed_at', ''), reverse=True)
        
        for task in completed_sorted:
            task_name = task.get('name', 'Unknown')
            timestamp = task.get('timestamp')
            
            with st.container():
                st.markdown(f"### ✅ {task_name}")
                cols = st.columns([3, 1])
                with cols[0]:
                    with st.expander("View Details", expanded=False):
                        st.json(task, expanded=False)
                    if task.get('completed_at'):
                        st.caption(f"⏰ Completed: {task.get('completed_at')}")
                    
                    if timestamp:
                        result_path = Path(RESULT_DIR) / timestamp
                        if result_path.exists():
                            result_files = list(result_path.glob('*.json'))
                            if result_files:
                                st.caption(f"📁 {len(result_files)} result file(s)")
                
                with cols[1]:
                    if timestamp:
                        result_path = Path(RESULT_DIR) / timestamp
                        if result_path.exists():
                            # Action buttons
                            if st.button("🔁 Retry", key=f"retry_completed_{task_name}", use_container_width=True):
                                tasks = load_tasks()
                                task_copy = next((t for t in tasks.get('completed', []) if t.get('name') == task_name), None)
                                if task_copy:
                                    # Remove status fields
                                    for key in ['started_at', 'completed_at', 'timestamp', 'stopped_at']:
                                        task_copy.pop(key, None)
                                    
                                    if move_task(tasks, task_name, 'completed', 'pending'):
                                        save_tasks(tasks)
                                        st.cache_data.clear()
                                        st.success("✅ Task moved to pending")
                                        time.sleep(0.5)
                                        st.rerun()
                            
                            log_tab, res_tab = st.tabs(["Logs", "Results"])
                            
                            with log_tab:
                                if st.button("📄 View", key=f"logs_{task_name}", use_container_width=True):
                                    log_files = {
                                        'Train': result_path / 'train.log',
                                        'Test': result_path / 'test.log',
                                        'Visualization': result_path / 'visualization.log'
                                    }
                                    tabs = st.tabs(["Train", "Test", "Visualization"])
                                    for tab, log_file in zip(tabs, log_files.values()):
                                        with tab:
                                            if log_file.exists():
                                                with open(log_file, 'r') as f:
                                                    st.text_area("", f.read(), height=300, key=f"log_{task_name}_{log_file.name}")
                                            else:
                                                st.info("Log not found")
                            
                            with res_tab:
                                if st.button("📊 View", key=f"results_{task_name}", use_container_width=True):
                                    train_json = result_path / f'train_{timestamp}.json'
                                    test_json = result_path / f'test_{timestamp}.json'
                                    
                                    res_tabs = st.tabs(["Training", "Testing"])
                                    
                                    with res_tabs[0]:
                                        if train_json.exists():
                                            with open(train_json, 'r') as f:
                                                st.json(json.load(f))
                                        else:
                                            st.info("Training results not found")
                                    
                                    with res_tabs[1]:
                                        if test_json.exists():
                                            with open(test_json, 'r') as f:
                                                test_data = json.load(f)
                                                st.json(test_data)
                                                if 'test_metrics' in test_data:
                                                    st.subheader("Metrics")
                                                    metrics = test_data['test_metrics']
                                                    metric_cols = st.columns(len(metrics))
                                                    for idx, (metric, value) in enumerate(metrics.items()):
                                                        metric_cols[idx].metric(metric, f"{value:.4f}")
                                        else:
                                            st.info("Test results not found")
                st.divider()
    else:
        st.info("✨ No completed tasks")

# Failed tasks
with tab4:
    failed = tasks.get('failed', [])
    if failed:
        for task in failed:
            task_name = task.get('name', 'Unknown')
            with st.container():
                st.markdown(f"### ❌ {task_name}")
                col1, col2 = st.columns([3, 1])
                with col1:
                    with st.expander("View Details", expanded=False):
                        st.json(task, expanded=False)
                    if task.get('error'):
                        st.error(f"**Error:** {task['error']}")
                    if task.get('completed_at'):
                        st.caption(f"⏰ Failed at: {task.get('completed_at')}")
                with col2:
                    # Retry button
                    if st.button("🔁 Retry", key=f"retry_failed_{task_name}", use_container_width=True, type="primary"):
                        tasks = load_tasks()
                        # Get task copy and clean status fields
                        task_copy = next((t for t in tasks.get('failed', []) if t.get('name') == task_name), None)
                        if task_copy:
                            # Remove status fields
                            for key in ['started_at', 'completed_at', 'error', 'timestamp', 'stopped_at']:
                                task_copy.pop(key, None)
                            
                            if move_task(tasks, task_name, 'failed', 'pending'):
                                save_tasks(tasks)
                                st.cache_data.clear()
                                st.success("✅ Task moved to pending")
                                time.sleep(0.5)
                                st.rerun()
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{task_name}", use_container_width=True):
                        tasks = load_tasks()
                        tasks['failed'] = [t for t in tasks.get('failed', []) if t.get('name') != task_name]
                        save_tasks(tasks)
                        st.cache_data.clear()
                        st.success("✅ Task deleted")
                        time.sleep(0.5)
                        st.rerun()
                st.divider()
    else:
        st.info("✨ No failed tasks")

# Add task
with tab5:
    st.subheader("➕ Create New Task")
    
    with st.form("add_task_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Settings")
            name = st.text_input("Task Name *", value="Task_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            batch_size = st.number_input("Batch Size", value=64, min_value=1, step=8)
            epochs = st.number_input("Epochs", value=200, min_value=1)
            lr = st.number_input("Learning Rate", value=0.0005, format="%.6f", step=0.0001)
            patience = st.number_input("Patience", value=30, min_value=1)
        
        with col2:
            st.markdown("#### Model Settings")
            loss_type = st.selectbox("Loss Type", ['ce', 'emd', 'focal', 'thr'], index=0)
            model_type = st.selectbox("Model Type", ['LSTM', 'CNN', 'Linear'], index=0)
            threshold_mode = st.selectbox("Threshold Mode", ['binary', 'value'], index=0)
            d_model = st.number_input("Hidden Dimension", value=128, min_value=1, step=32)
            dropout = st.number_input("Dropout", value=0.1, min_value=0.0, max_value=1.0, step=0.05)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Training Settings")
            lr_scheduler = st.selectbox("LR Scheduler", ['none', 'cosine', 'plateau'], index=1)
            warmup_epochs = st.number_input("Warmup Epochs", value=5, min_value=0)
            grad_clip = st.number_input("Gradient Clip", value=1.0, min_value=0.0, step=0.1)
        
        with col4:
            st.markdown("#### Loss Settings")
            use_weighted_loss = st.checkbox("Use Weighted Loss", value=True)
            pos_weight = st.number_input("Positive Weight", value=5.0, min_value=0.1, step=0.5)
            metrics_threshold = st.number_input("Metrics Threshold", value=0.5, min_value=0.0, max_value=1.0, step=0.05)
        
        submitted = st.form_submit_button("➕ Add Task", use_container_width=True, type="primary")
        
        if submitted:
            if not name or not name.strip():
                st.error("Task name is required!")
            else:
                new_task = {
                    'name': name.strip(),
                    'batch_size': int(batch_size),
                    'epochs': int(epochs),
                    'lr': float(lr),
                    'patience': int(patience),
                    'loss_type': loss_type,
                    'model_type': model_type,
                    'threshold_mode': threshold_mode,
                    'd_model': int(d_model),
                    'dropout': float(dropout),
                    'use_weighted_loss': use_weighted_loss,
                    'pos_weight': float(pos_weight) if use_weighted_loss else 5.0,
                    'lr_scheduler': lr_scheduler,
                    'warmup_epochs': int(warmup_epochs),
                    'metrics_threshold': float(metrics_threshold),
                    'grad_clip': float(grad_clip),
                    'result_dir': 'result'
                }
                tasks = load_tasks()
                tasks.setdefault('pending', []).append(new_task)
                save_tasks(tasks)
                st.cache_data.clear()
                st.success(f"✅ Task '{name}' added!")
                time.sleep(0.5)
                st.rerun()

# Sidebar
with st.sidebar:
    st.header("⚙️ Control")
    
    st.markdown("### Auto Flow")
    if st.session_state.auto_flow_running:
        st.warning("🔄 Running")
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.auto_flow_running = False
            st.success("✅ Stopped")
            st.rerun()
    else:
        if st.button("🚀 Start Auto Flow", type="primary", use_container_width=True):
            try:
                st.session_state.auto_flow_running = True
                thread = threading.Thread(target=run_flow_background, daemon=True)
                thread.start()
                st.success("✅ Started!")
            except Exception as e:
                st.session_state.auto_flow_running = False
                st.error(f"❌ {e}")
    
    st.divider()
    
    st.markdown("### Statistics")
    stats = {
        'Pending': len(tasks.get('pending', [])),
        'Running': len(tasks.get('running', [])),
        'Completed': len(tasks.get('completed', [])),
        'Failed': len(tasks.get('failed', [])),
        'Cancelled': len(tasks.get('cancelled', []))
    }
    st.bar_chart(stats)
    
    st.divider()
    
    st.markdown("### Actions")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("🧹 Clean Empty Dirs", use_container_width=True):
        with st.spinner("Cleaning..."):
            cleaned = cleanup_empty_dirs(RESULT_DIR)
            st.success(f"✅ Cleaned {cleaned} empty directories")
            time.sleep(1)
            st.rerun()
    
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
