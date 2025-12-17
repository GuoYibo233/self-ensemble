# Generate Scripts - Usage Guide

本项目包含四个不同的生成脚本，每个脚本实现了不同的生成方法。本文档介绍所有generate文件的区别和使用方法。

## 📋 脚本概览

| 脚本 | 用途 | 数据集 | 方法类型 |
|------|------|--------|---------|
| `generate_baseline.py` | 基准测试 | WebQA | 单独生成（origin/per_prompt） |
| `generate_original.py` | 原始集成方法 | WebQA | Logit级融合（max/avg/weighted） |
| `generate_flex_attention.py` | FlexAttention集成 | WebQA | Attention级融合 |
| `generate_myriadlama.py` | MyriadLAMA特定方法 | MyriadLAMA | FlexAttention（针对填空任务） |

---

## 1. generate_baseline.py - 基准生成

**目的**: 为集成方法提供基准对比结果

**数据集**: WebQA

**支持的方法**:
- `origin`: 仅使用原始问题（基准1）
- `per_prompt`: 每个释义单独生成（基准2）

### 使用方法

```bash
# 基准1: 仅原始问题
python src/generate_baseline.py \
    --method origin \
    --dataset webqa \
    --model llama3.2_3b_it

# 基准2: 每个释义单独生成
python src/generate_baseline.py \
    --method per_prompt \
    --dataset webqa \
    --model llama3.2_3b_it

# 生成所有基准
python src/generate_baseline.py \
    --method all \
    --dataset webqa \
    --model llama3.2_3b_it
```

---

## 2. generate_original.py

**目的**: 自动生成paraphase再集成，使用的是webqa数据集

**数据集**: WebQA

**支持的方法**:
- `max`: 在每一步选择最大logit
- `avg`: 对所有logits求平均
- `weighted_avg`: 基于置信度的加权平均
- `weighted_max`: 基于置信度的加权最大值

### 使用方法

```bash
# 最大值集成
python src/generate_original.py \
    --method max \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6

# 平均值集成
python src/generate_original.py \
    --method avg \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6

# 加权平均集成
python src/generate_original.py \
    --method weighted_avg \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_ensemble 6
```


---

## 3. generate_flex_attention.py - FlexAttention集成

**目的**: 使用FlexAttention实现高效的attention级集成

**数据集**: myraidlama

**方法**: FlexAttention - 在单次前向传播中融合多个释义

### 使用方法

```bash
# FlexAttention集成（5个释义）
python src/generate_flex_attention.py \
    --dataset myraidlama \
    --model llama3.2_3b_it \
    --num_paraphrases 5

# 限制样本数量（快速测试）
python src/generate_flex_attention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5 \
    --max_samples 100
```

### 工作原理

1. **拼接**: 一起拼的
   ```
   [ins fewshot paraphrase1] [ins fewshot paraphrase2] [ins fewshot paraphrase3] [ins fewshot paraphrase4] [ins fewshot paraphrase5]
   ```

2. **隔离编码**: 每个释义在编码阶段互不干扰
   ```
   Para1: ✓✓✓ ✗✗✗ ✗✗✗ ✗✗✗ ✗✗✗
   Para2: ✗✗✗ ✓✓✓ ✗✗✗ ✗✗✗ ✗✗✗
   Para3: ✗✗✗ ✗✗✗ ✓✓✓ ✗✗✗ ✗✗✗
   ```

3. **融合生成**: 生成的token可以关注所有释义
   ```
   Gen1: ✓✓✓ ✓✓✓ ✓✓✓ ✓✓✓ ✓✓✓
   Gen2: ✓✓✓ ✓✓✓ ✓✓✓ ✓✓✓ ✓✓✓ ✓
   ```



### 环境要求

```bash
# 安装PyTorch nightly（支持FlexAttention）
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

---

## 4. generate_myriadlama.py - MyriadLAMA特定生成

**目的**: 只使用myraidlama

**数据集**: MyriadLAMA

**方法**: FlexAttention

### 使用方法

```bash
# MyriadLAMA FlexAttention生成
python src/generate_myriadlama.py \
    --dataset myriadlama \
    --model llama3.2_3b_it \
    --num_paraphrases 5
```
---

## 📦 依赖文件说明

所有generate脚本依赖以下核心模块：

### 核心模块（src/core/）
- `constants.py` - 模型路径配置
- `dataset.py` - 数据集加载器
- `paraphrase.py` - 释义生成
- `confidence.py` - 置信度计算
- `utils.py` - 通用工具函数
- `interactive.py` - 交互式参数输入

### 根目录模块
- `mask_visualization.py` - Attention mask可视化（仅flex_attention和myriadlama使用）

---


---

## 🗂️ 文件结构

```
.
├── src/
│   ├── core/                      # 核心模块
│   │   ├── constants.py           # 配置
│   │   ├── dataset.py             # 数据集
│   │   ├── paraphrase.py          # 释义
│   │   ├── confidence.py          # 置信度
│   │   ├── utils.py               # 工具
│   │   └── interactive.py         # 交互
│   │
│   ├── generate_baseline.py       # 基准生成
│   ├── generate_original.py       # 原始集成
│   ├── generate_flex_attention.py # FlexAttention
│   ├── generate_myriadlama.py     # MyriadLAMA
│   └── run_interactive.py         # 交互式运行
│
├── mask_visualization.py          # Mask可视化
├── requirements.txt               # Python依赖
├── environment.yml                # Conda环境
└── archived/                      # 归档文件
    ├── docs/                      # 详细文档
    ├── tests/                     # 测试文件
    ├── tools/                     # 工具脚本
    └── ...                        # 其他归档文件
```

---