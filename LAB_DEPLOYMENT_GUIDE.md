# 实验室MoE模型训练部署指南

## 🎯 概述
这是一套完整的MoE (Mixture of Experts) 模型训练框架，专为实验室服务器环境设计。基于之前成功的AVG自组装方法（在250样本上达到42.8%准确率），现在扩展到MoE模型以获得更好的性能。

## 🔧 环境准备

### 1. 服务器要求
- **GPU**: 至少1张RTX 3090/4090或A100 (建议多GPU)
- **内存**: 32GB+ RAM
- **存储**: 500GB+ 可用空间
- **CUDA**: 11.8+

### 2. 快速部署
```bash
# 1. 克隆代码到服务器
git clone <your-repo> /data/self-ensemble-moe
cd /data/self-ensemble-moe

# 2. 运行部署脚本
chmod +x deploy_to_lab_server.sh
./deploy_to_lab_server.sh

# 3. 验证环境
python test_environment.py
```

## 🚀 开始训练

### 方法1: 自动实验套件
```bash
# 运行所有MoE实验
chmod +x scripts/run_moe_experiments.sh
./scripts/run_moe_experiments.sh
```

### 方法2: 手动运行单个实验
```bash
# 激活环境
conda activate self-ensemble-moe

# 运行Qwen1.5-MoE-A2.7B-Chat
python train_moe_lab.py \
    --model_name qwen1.5_moe_a2.7b_chat \
    --dataset myriadlama \
    --max_samples 500 \
    --batch_size 4 \
    --experiment_name my_moe_experiment \
    --output_dir /data/results
```

## 📊 监控训练

### 实时监控
```bash
# 启动监控 (每30秒刷新)
python monitor_moe_training.py --action monitor --interval 30

# 保存监控日志
python monitor_moe_training.py --action monitor --log-file /data/logs/monitor.log
```

### 检查训练状态
```bash
# 查看所有实验状态
python monitor_moe_training.py --action status

# 查看GPU使用情况
nvidia-smi

# 查看运行中的进程
ps aux | grep train_moe_lab
```

## 🔧 配置说明

### GPU配置推荐
- **RTX 3090 (24GB)**: batch_size=4, max_samples=500
- **RTX 4090 (24GB)**: batch_size=6, max_samples=750  
- **A100 (40GB+)**: batch_size=8, max_samples=1000

### 模型选择策略
1. **Qwen1.5-MoE-A2.7B-Chat**: 最推荐，专为对话优化
2. **Qwen1.5-MoE-A2.7B**: 基础版本，可用于对比
3. **Mixtral-8x7B**: 大模型，需要更多GPU内存

## 📁 目录结构
```
/data/self-ensemble-moe/
├── train_moe_lab.py           # 主训练脚本
├── lab_moe_config.py          # 实验室配置
├── monitor_moe_training.py    # 监控脚本
├── scripts/
│   └── run_moe_experiments.sh # 批量实验脚本
├── results/                   # 结果输出
└── logs/                     # 训练日志
```

## 🐛 常见问题

### Q1: CUDA out of memory
**解决方案**:
```bash
# 减少batch size
python train_moe_lab.py --batch_size 2

# 或使用梯度累积
python train_moe_lab.py --gradient_accumulation_steps 2
```

### Q2: 模型下载失败
**解决方案**:
```bash
# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地
huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B-Chat
```

### Q3: 多GPU使用
**解决方案**:
```bash
# 设置可见GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 使用多GPU训练
python train_moe_lab.py --use_multi_gpu
```

## 📈 性能基准

### 预期结果 (基于MyriadLAMA数据集)
- **Qwen2.5-1.5B + AVG**: 42.8% (已验证)
- **Qwen1.5-MoE-A2.7B + AVG**: 预期 45-50%
- **Mixtral-8x7B + AVG**: 预期 50-55%

### 训练时间估算
- **500样本**: ~30-60分钟 (单GPU)
- **1000样本**: ~1-2小时 (单GPU)
- **使用多GPU可显著加速**

## 🔗 相关文件
- `constants.py`: 模型路径配置
- `moe_config.py`: MoE理论分析和建议
- `utils.py`: 工具函数
- `dataset.py`: 数据集处理

## 💡 优化建议

1. **数据预处理**: 使用SSD存储数据集
2. **模型缓存**: 预下载模型到本地
3. **批量实验**: 使用脚本进行批量测试
4. **结果分析**: 定期备份实验结果

## 📞 支持
如果遇到问题，请检查：
1. GPU内存是否充足
2. CUDA版本是否兼容
3. 网络连接是否正常
4. 磁盘空间是否充足

---
📝 **更新日志**:
- 2024-12-19: 初始版本，基于成功的AVG方法扩展
- 预期下次更新: 添加更多MoE模型支持