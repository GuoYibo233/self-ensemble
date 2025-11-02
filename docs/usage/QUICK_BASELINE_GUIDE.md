# 批量生成Baseline脚本 - 快速指南

## 🚀 快速开始

### 方式1: 使用Bash脚本（推荐）

```bash
cd /home/y-guo/self-ensemble/self-ensemble

# 1. 查看会做什么（不实际执行）
bash scripts/generate_all_baselines.sh --dry-run

# 2. 执行生成
bash scripts/generate_all_baselines.sh

# 3. 如果中断了，重新运行即可继续（会自动跳过已完成的）
bash scripts/generate_all_baselines.sh
```

### 方式2: 使用Python脚本（更多选项）

```bash
cd /home/y-guo/self-ensemble/self-ensemble

# 1. 查看会做什么
python3 scripts/generate_all_baselines.py --dry-run

# 2. 执行生成
python3 scripts/generate_all_baselines.py

# 3. 只处理WebQA数据集
python3 scripts/generate_all_baselines.py --dataset webqa
```

## 📋 当前需要处理的模型

根据刚才的扫描，需要为以下9个模型生成baseline：

### WebQA (8个模型)
1. llama3.2_1b
2. llama3.2_1b_it
3. llama3.2_3b
4. llama3.2_3b_it
5. qwen2.5_7b_it
6. qwen3_1.7b
7. qwen3_4b
8. qwen3_8b

### MyriadLAMA (1个模型)
9. qwen1.5_moe_a2.7b_chat

## ⏱️ 预计时间

- **单个WebQA模型**: 35-75分钟
- **所有9个模型**: 约6-12小时（顺序执行）

## 🔧 推荐运行方式

### 在tmux中后台运行

```bash
# 1. 创建tmux session
tmux new -s baseline_gen

# 2. 在tmux中运行脚本
cd /home/y-guo/self-ensemble/self-ensemble
bash scripts/generate_all_baselines.sh

# 3. 分离tmux（脚本继续在后台运行）
# 按 Ctrl+B 然后按 D

# 4. 稍后重新连接查看进度
tmux attach -t baseline_gen

# 5. 或在另一个终端监控进度
watch -n 60 'ls -lh /net/tokyo100-10g/data/str01_01/y-guo/datasets/webqa/*/baseline_*.feather | tail -20'
```

## 📊 脚本功能

### 自动化功能
- ✅ 自动扫描所有已有模型
- ✅ 检测哪些模型已有baseline（自动跳过）
- ✅ 按顺序处理每个模型
- ✅ 显示详细进度和统计
- ✅ 支持中断后继续

### 每个模型生成两个文件
1. **baseline_origin.feather** - 只用原始问题
2. **baseline_per_prompt.feather** - 每个paraphrase单独生成

## 📁 生成的文件位置

```
/net/tokyo100-10g/data/str01_01/y-guo/datasets/
├── webqa/
│   ├── llama3.2_1b/
│   │   ├── paraphrases_dataset/
│   │   ├── baseline_origin.feather       ← 新生成
│   │   └── baseline_per_prompt.feather   ← 新生成
│   ├── llama3.2_3b_it/
│   │   ├── paraphrases_dataset/
│   │   ├── baseline_origin.feather       ← 新生成
│   │   └── baseline_per_prompt.feather   ← 新生成
│   └── ...
│
└── myriadlama/
    └── qwen1.5_moe_a2.7b_chat/
        ├── baseline_origin.feather       ← 新生成
        └── baseline_per_prompt.feather   ← 新生成
```

## 🔍 监控进度

### 方法1: 查看文件生成
```bash
# 查看已生成的baseline文件
ls -lh /net/tokyo100-10g/data/str01_01/y-guo/datasets/webqa/*/baseline_*.feather

# 实时监控
watch -n 60 'ls -lh /net/tokyo100-10g/data/str01_01/y-guo/datasets/webqa/*/baseline_*.feather | wc -l'
```

### 方法2: 监控GPU
```bash
watch -n 1 nvidia-smi
```

### 方法3: 查看日志
如果在tmux中运行，直接连接到session查看输出。

## ⚠️ 注意事项

1. **不会重复生成**: 已存在的baseline会被自动跳过
2. **可以中断**: 按Ctrl+C中断后，下次运行会从未完成的继续
3. **GPU占用**: 每次只运行一个模型，避免资源冲突
4. **磁盘空间**: 确保有足够空间（每个模型约2-5GB）

## 🎯 执行命令

### 立即开始生成所有baseline

```bash
# 进入项目目录
cd /home/y-guo/self-ensemble/self-ensemble

# 创建tmux session
tmux new -s baseline_gen

# 运行脚本
bash scripts/generate_all_baselines.sh

# 分离tmux: Ctrl+B 然后按 D
```

### 或者只生成WebQA的baseline

```bash
cd /home/y-guo/self-ensemble/self-ensemble
tmux new -s baseline_gen
python3 scripts/generate_all_baselines.py --dataset webqa
# Ctrl+B D
```

## 📈 完成后

生成完成后可以：

1. **分析单个模型**:
   ```bash
   python analysis/analyze_baseline.py --dataset webqa --model llama3.2_3b_it
   ```

2. **对比所有模型**:
   ```bash
   python analysis/compare_all_baselines.py
   ```

3. **生成报告**:
   ```bash
   python analysis/generate_baseline_report.py
   ```

---

**准备好了吗？运行上面的命令开始生成！** 🚀
