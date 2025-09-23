#!/usr/bin/env python3
"""
实验室MoE训练监控脚本
monitor_moe_training.py
"""

import time
import psutil
import subprocess
import json
import os
from datetime import datetime
from pathlib import Path
import argparse


def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': int(parts[2]),
                    'memory_total': int(parts[3]),
                    'utilization': int(parts[4]),
                    'temperature': int(parts[5])
                })
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []


def get_process_info():
    """获取MoE训练进程信息"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if 'train_moe_lab.py' in ' '.join(proc.info['cmdline']):
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                    'cmdline': ' '.join(proc.info['cmdline'][:5])  # 只显示前5个参数
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def monitor_training(interval=30, log_file=None):
    """监控训练过程"""
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    print("🔍 Starting MoE Training Monitor")
    print("=" * 60)
    print(f"Monitoring interval: {interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 60)

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 获取系统信息
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # 获取GPU信息
            gpus = get_gpu_info()

            # 获取进程信息
            processes = get_process_info()

            # 构建监控数据
            monitor_data = {
                'timestamp': timestamp,
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / 1024**3,
                    'disk_percent': disk.percent
                },
                'gpus': gpus,
                'processes': processes
            }

            # 输出到终端
            print(f"\n📊 [{timestamp}]")
            print(
                f"CPU: {cpu_percent:.1f}% | Memory: {memory.percent:.1f}% | Disk: {disk.percent:.1f}%")

            for gpu in gpus:
                memory_percent = (gpu['memory_used'] /
                                  gpu['memory_total']) * 100
                print(f"GPU {gpu['index']} ({gpu['name']}): "
                      f"Util: {gpu['utilization']}% | "
                      f"Memory: {memory_percent:.1f}% ({gpu['memory_used']}MB/{gpu['memory_total']}MB) | "
                      f"Temp: {gpu['temperature']}°C")

            if processes:
                print("🔄 Training Processes:")
                for proc in processes:
                    print(f"  PID {proc['pid']}: CPU {proc['cpu_percent']:.1f}% | "
                          f"RAM {proc['memory_mb']:.0f}MB | {proc['cmdline']}")
            else:
                print("⚠️  No training processes found")

            # 保存到日志文件
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(json.dumps(monitor_data) + '\n')

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")


def check_training_status(results_dir):
    """检查训练状态"""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print("📋 Training Status Summary")
    print("=" * 60)

    experiments = []
    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir():
            config_file = exp_dir / 'config.json'
            results_file = exp_dir / 'results.json'

            status = "Unknown"
            accuracy = None

            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

                if results_file.exists():
                    with open(results_file) as f:
                        results = json.load(f)
                    status = "Completed"
                    accuracy = results.get('accuracy', 0)
                else:
                    # 检查是否正在运行
                    processes = get_process_info()
                    if any(config['experiment_name'] in proc['cmdline'] for proc in processes):
                        status = "Running"
                    else:
                        status = "Failed/Interrupted"

                experiments.append({
                    'name': config['experiment_name'],
                    'model': config['model_name'],
                    'status': status,
                    'accuracy': accuracy
                })

    if not experiments:
        print("No experiments found")
        return

    for exp in experiments:
        acc_str = f" (Acc: {exp['accuracy']:.3f})" if exp['accuracy'] else ""
        print(
            f"{exp['name']:30} | {exp['model']:20} | {exp['status']:15}{acc_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor MoE training on lab server')
    parser.add_argument('--action', choices=['monitor', 'status'], default='monitor',
                        help='Action to perform')
    parser.add_argument('--interval', type=int, default=30,
                        help='Monitoring interval in seconds')
    parser.add_argument('--log-file', type=str,
                        help='Log file to save monitoring data')
    parser.add_argument('--results-dir', type=str, default='/data/results/moe-experiments',
                        help='Results directory to check status')

    args = parser.parse_args()

    if args.action == 'monitor':
        monitor_training(args.interval, args.log_file)
    elif args.action == 'status':
        check_training_status(args.results_dir)


if __name__ == '__main__':
    main()
