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

### 特点
- ✅ 最简单的生成方法
- ✅ 不使用任何集成技术
- ✅ 适合作为对比基准

---

## 2. generate_original.py - 原始集成方法

**目的**: 实现传统的logit级集成方法

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

### 特点
- ✅ 传统的集成方法
- ✅ Logit级别的融合
- ❌ 每一步需要N次前向传播（效率较低）
- ✅ 支持多种融合策略

---

## 3. generate_flex_attention.py - FlexAttention集成

**目的**: 使用FlexAttention实现高效的attention级集成

**数据集**: WebQA

**方法**: FlexAttention - 在单次前向传播中融合多个释义

### 使用方法

```bash
# FlexAttention集成（5个释义）
python src/generate_flex_attention.py \
    --dataset webqa \
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

1. **拼接**: 将5个释义拼接成一个提示
   ```
   Para1 [SEP] Para2 [SEP] Para3 [SEP] Para4 [SEP] Para5
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

### 特点
- ✅ **最高效**: 每步仅需1次前向传播（vs N次）
- ✅ Attention级别的融合
- ✅ 与logit级方法质量相当或更好
- ⚠️ 需要PyTorch 2.5+或nightly版本

### 环境要求

```bash
# 安装PyTorch nightly（支持FlexAttention）
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

---

## 4. generate_myriadlama.py - MyriadLAMA特定生成

**目的**: 为MyriadLAMA数据集的填空任务优化的FlexAttention方法

**数据集**: MyriadLAMA

**方法**: FlexAttention（针对填空任务优化）

### 使用方法

```bash
# MyriadLAMA FlexAttention生成
python src/generate_myriadlama.py \
    --dataset myriadlama \
    --model llama3.2_3b_it \
    --num_paraphrases 5
```

### 与generate_flex_attention.py的区别

| 特性 | generate_flex_attention.py | generate_myriadlama.py |
|------|---------------------------|------------------------|
| **数据集** | WebQA（问答） | MyriadLAMA（填空） |
| **任务类型** | 长文本生成 | 单词预测 |
| **提示格式** | 标准问答格式 | [MASK]填空格式 |
| **Mask逻辑** | 释义隔离 | 释义+Few-shot样例隔离 |
| **Few-shot** | 标准few-shot | 每个样例独立隔离 |
| **输出长度** | 可变长度 | 通常单个token |

### 特点
- ✅ 专为MyriadLAMA优化
- ✅ 支持[MASK]填空任务
- ✅ Few-shot样例之间互相隔离
- ✅ 针对单词预测优化

---

## 🔄 方法对比总结

### 效率对比

| 方法 | 每步前向传播次数 | 相对速度 | 融合级别 |
|------|-----------------|---------|---------|
| baseline (origin) | 1× | 最快 | 无融合 |
| baseline (per_prompt) | N× | 标准 | 无融合 |
| original (max/avg) | N× | 标准 | Logit级 |
| flex_attention | 1× | **最快** | **Attention级** |
| myriadlama | 1× | **最快** | **Attention级** |

### 质量对比

| 方法 | 准确性 | 多样性 | 适用场景 |
|------|--------|--------|---------|
| baseline (origin) | 基准 | 低 | 对比基准 |
| baseline (per_prompt) | 中等 | 高 | 对比基准 |
| original (max/avg) | 高 | 中等 | 通用问答 |
| flex_attention | **高** | 高 | 通用问答（推荐） |
| myriadlama | **高** | 高 | 填空任务 |

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

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建conda环境
conda env create -f environment.yml
conda activate flexattention

# 或使用Linux特定环境
conda env create -f environment_linux.yml
conda activate self-ensemble-debug
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 3. 运行生成

```bash
# 推荐：使用交互式模式
python src/run_interactive.py

# 或直接运行特定脚本
python src/generate_flex_attention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5
```

---

## 📊 选择合适的脚本

### 根据任务类型选择

- **WebQA问答任务** → `generate_flex_attention.py` （推荐）
- **MyriadLAMA填空任务** → `generate_myriadlama.py`
- **需要对比基准** → `generate_baseline.py`
- **研究不同融合方法** → `generate_original.py`

### 根据效率要求选择

- **最快速度** → `generate_flex_attention.py` 或 `generate_myriadlama.py`
- **标准速度，多种方法** → `generate_original.py`
- **简单基准** → `generate_baseline.py`

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

## 📝 注意事项

1. **FlexAttention脚本**（`generate_flex_attention.py`和`generate_myriadlama.py`）需要PyTorch 2.5+或nightly版本
2. 所有脚本都支持`--max_samples`参数用于快速测试
3. 使用`--help`查看每个脚本的完整参数列表
4. 归档的文档（`archived/docs/`）包含更详细的技术说明

---

## 🔗 相关链接

- 主README: [README.md](README.md)
- 归档文档: [archived/docs/](archived/docs/)
- 交互式运行: `python src/run_interactive.py`

---

**最后更新**: 2025-12-17
