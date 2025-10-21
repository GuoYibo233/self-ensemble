# Baseline Generation Scripts

这个目录包含用于批量生成baseline的脚本。

## 📁 脚本说明

### 1. `generate_all_baselines.sh` - Bash脚本版本

简单的bash脚本，自动扫描所有已有模型并生成baseline。

**特点**:
- ✅ 简单易用
- ✅ 自动跳过已存在的baseline
- ✅ 彩色输出
- ✅ 完整的统计信息

**用法**:
```bash
# 基本用法（扫描并生成所有缺失的baseline）
bash scripts/generate_all_baselines.sh

# Dry run（查看会做什么，不实际执行）
bash scripts/generate_all_baselines.sh --dry-run

# 强制重新生成所有baseline
bash scripts/generate_all_baselines.sh --rewrite
```

### 2. `generate_all_baselines.py` - Python脚本版本

功能更强大的Python版本，支持更多选项。

**特点**:
- ✅ 自动扫描和检测
- ✅ 显示已有baseline状态
- ✅ 支持选择特定数据集
- ✅ 交互式确认
- ✅ 详细的进度显示
- 🚧 未来支持并行执行

**用法**:
```bash
# 基本用法
python scripts/generate_all_baselines.py

# Dry run
python scripts/generate_all_baselines.py --dry-run

# 只处理WebQA
python scripts/generate_all_baselines.py --dataset webqa

# 强制重新生成
python scripts/generate_all_baselines.py --rewrite

# 指定数据集根目录
python scripts/generate_all_baselines.py --dataset-root /path/to/datasets
```

## 🎯 工作流程

### 步骤1: 查看现有模型

```bash
# 使用bash脚本dry run
bash scripts/generate_all_baselines.sh --dry-run

# 或使用Python脚本
python scripts/generate_all_baselines.py --dry-run
```

**输出示例**:
```
======================================================================
Scanning for Existing Models
======================================================================

WEBQA:
  - llama3.2_1b              [origin✗ per_prompt✗]
  - llama3.2_1b_it          [origin✗ per_prompt✗]
  - llama3.2_3b              [origin✗ per_prompt✗]
  - llama3.2_3b_it          [origin✓ per_prompt✓]  <- 已有baseline
  - qwen2.5_3b_it           [origin✗ per_prompt✗]
  - qwen2.5_7b_it           [origin✗ per_prompt✗]
  - qwen3_1.7b               [origin✗ per_prompt✗]
  - qwen3_4b                 [origin✗ per_prompt✗]
  - qwen3_8b                 [origin✗ per_prompt✗]

MYRIADLAMA:
  - qwen1.5_moe_a2.7b_chat  [origin✗ per_prompt✗]

Total models found: 10
Models to process: 9
```

### 步骤2: 生成所有baseline

```bash
# 推荐：使用bash脚本（更稳定）
bash scripts/generate_all_baselines.sh

# 或使用Python脚本
python scripts/generate_all_baselines.py
```

脚本会：
1. 扫描所有已有模型目录
2. 检查哪些模型缺少baseline
3. 逐个生成baseline（origin + per_prompt）
4. 显示进度和统计信息

### 步骤3: 查看结果

生成完成后，每个模型目录下会有：
```
/net/.../datasets/webqa/llama3.2_3b_it/
├── paraphrases_dataset/
├── baseline_origin.feather       # 新生成
└── baseline_per_prompt.feather   # 新生成
```

## 📊 预计运行时间

对于WebQA数据集（~1943个问题）：

| 模型 | Baseline 1 (origin) | Baseline 2 (per_prompt) | 总计 |
|------|---------------------|-------------------------|------|
| llama3.2_3b_it | ~5-10分钟 | ~30-40分钟 | ~35-50分钟 |
| qwen2.5_7b_it | ~8-15分钟 | ~45-60分钟 | ~53-75分钟 |

**所有9个模型**: 约6-12小时（顺序执行）

## 🔧 高级用法

### 只重新生成特定数据集

```bash
# 只处理WebQA
python scripts/generate_all_baselines.py --dataset webqa

# 只处理MyriadLAMA
python scripts/generate_all_baselines.py --dataset myriadlama
```

### 在tmux中后台运行

```bash
# 创建新的tmux session
tmux new -s baseline_gen

# 在tmux中运行脚本
bash scripts/generate_all_baselines.sh

# 按 Ctrl+B 然后按 D 分离session
# 稍后重新连接: tmux attach -t baseline_gen
```

### 监控进度

```bash
# 在另一个终端监控输出文件生成
watch -n 60 'ls -lh /net/tokyo100-10g/data/str01_01/y-guo/datasets/webqa/*/baseline_*.feather'

# 或监控GPU使用
watch -n 1 nvidia-smi
```

## 🐛 故障排除

### 问题1: 某个模型失败

**症状**: 脚本在某个模型处失败并退出

**解决**:
```bash
# 跳过失败的模型，手动为其生成baseline
python baseline_generate.py --method all --dataset webqa --model 失败的模型名

# 然后重新运行脚本继续处理其他模型
bash scripts/generate_all_baselines.sh
```

### 问题2: 内存不足

**症状**: CUDA out of memory

**解决**:
```bash
# 减少batch size（需要修改baseline_generate.py中的batch_size）
# 或一次只生成一个模型
python baseline_generate.py --method all --dataset webqa --model llama3.2_3b_it
```

### 问题3: 磁盘空间不足

**症状**: No space left on device

**解决**:
```bash
# 检查磁盘使用
df -h /net/tokyo100-10g/data/str01_01/y-guo/

# 清理不需要的旧结果
rm /net/.../datasets/webqa/old_model/*.feather
```

## 📝 脚本输出示例

```
========================================================================
Generate Baselines for All Existing Models
========================================================================
Dataset root: /net/tokyo100-10g/data/str01_01/y-guo/datasets
Project root: /home/y-guo/self-ensemble/self-ensemble
Rewrite: false
Dry run: false

========================================================================
Scanning WebQA Models
========================================================================

--------------------------------------------------------------------
Dataset: webqa | Model: llama3.2_1b
--------------------------------------------------------------------
Executing: python3 baseline_generate.py --method all --dataset webqa --model llama3.2_1b

======================================================================
Baseline 1: Origin (Attention Mode Baseline)
======================================================================
Method: Uses only original questions (no paraphrases)
Output: .../baseline_origin.feather

Generating baseline (origin): 100%|████████| 243/243 [05:23<00:00]

✅ Baseline 1 (origin) results saved to: baseline_origin.feather
   Total samples: 1943

======================================================================
Baseline 2: Per-Prompt (Attention Mode Second Baseline)
======================================================================
...

✅ Successfully generated baselines for webqa/llama3.2_1b

--------------------------------------------------------------------
Dataset: webqa | Model: llama3.2_3b
--------------------------------------------------------------------
...

========================================================================
Summary
========================================================================
Total models found: 10
Generated: 9
Skipped: 1
Done!
========================================================================
```

## 🎯 下一步

生成完所有baseline后：

1. **分析结果**:
   ```bash
   python analysis/analyze_baseline.py --dataset webqa --model llama3.2_3b_it
   ```

2. **对比所有模型**:
   ```bash
   python analysis/compare_all_baselines.py
   ```

3. **生成报告**:
   ```bash
   python analysis/generate_baseline_report.py --output baseline_report.md
   ```

## 📚 相关文档

- [BASELINE_USAGE.md](../BASELINE_USAGE.md) - Baseline使用指南
- [baseline_generate.py](../baseline_generate.py) - 单个模型的baseline生成脚本
- [analyze_baseline.py](../analysis/analyze_baseline.py) - Baseline分析脚本
