# Self-Ensemble with FlexAttention

本仓库实现了多种自集成（self-ensemble）文本生成方法，包括基于FlexAttention的高效注意力级融合方法。

## 📖 文档导航

**首次使用请阅读:**

1. **[GENERATE_README.md](GENERATE_README.md)** - 所有生成脚本的详细说明和区别（中文）
2. **[archived/docs/](archived/docs/)** - 完整的技术文档（已归档）

## 🎯 核心特性

本仓库提供**四种生成方法**，适用于不同的使用场景：

1. **Baseline生成** - 基准对比方法（origin/per_prompt）
2. **Original集成** - 传统logit级融合（max/avg/weighted）
3. **FlexAttention集成** - 高效的attention级融合（WebQA）
4. **MyriadLAMA集成** - 针对填空任务优化的FlexAttention

### 方法对比

| 方法 | 融合方式 | 效率 | 适用场景 |
|------|---------|------|---------|
| Baseline | 无融合 | 最快 | 对比基准 |
| Original | Logit级 | 标准 (N×前向) | 研究不同融合策略 |
| **FlexAttention** | **Attention级** | **最高效 (1×前向)** | **WebQA问答（推荐）** |
| MyriadLAMA | Attention级 | 最高效 (1×前向) | 填空任务 |

详细对比请参考：[GENERATE_README.md](GENERATE_README.md)

## 🔧 环境配置

### 系统要求

- Python 3.10+
- PyTorch 2.5+ 或 nightly（FlexAttention需要）
- NVIDIA GPU with CUDA
- Conda/Miniconda

### 快速安装

```bash
# 1. 创建conda环境
conda env create -f environment.yml
conda activate flexattention

# 或使用Linux特定环境（Ubuntu 22.04+）
conda env create -f environment_linux.yml
conda activate self-ensemble-debug

# 2. 安装依赖
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 3. 安装PyTorch nightly（支持FlexAttention）
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

详细配置说明：[archived/docs/QUICKSTART.md](archived/docs/QUICKSTART.md)

## 📖 使用方法

### 方式一：交互式运行（推荐）

```bash
python src/run_interactive.py
```

交互式界面会引导您选择：
- 生成类型（baseline/original/flex_attention/myriadlama）
- 数据集（webqa/myriadlama）
- 模型
- 方法特定参数

### 方式二：直接运行

```bash
# Baseline基准生成
python src/generate_baseline.py --method origin --dataset webqa --model llama3.2_3b_it

# Original集成方法
python src/generate_original.py --method max --dataset webqa --model llama3.2_3b_it --num_ensemble 6

# FlexAttention集成（推荐）
python src/generate_flex_attention.py --dataset webqa --model llama3.2_3b_it --num_paraphrases 5

# MyriadLAMA填空任务
python src/generate_myriadlama.py --dataset myriadlama --model llama3.2_3b_it --num_paraphrases 5
```

### 快速测试

```bash
# 限制样本数量，快速测试
python src/generate_flex_attention.py \
    --dataset webqa \
    --model llama3.2_3b_it \
    --num_paraphrases 5 \
    --max_samples 100
```

**完整使用指南**: [GENERATE_README.md](GENERATE_README.md)

## 📁 仓库结构

```
.
├── src/                           # 核心生成脚本
│   ├── core/                      # 共享模块
│   │   ├── constants.py           # 模型配置
│   │   ├── dataset.py             # 数据集加载
│   │   ├── paraphrase.py          # 释义生成
│   │   ├── confidence.py          # 置信度计算
│   │   ├── utils.py               # 工具函数
│   │   └── interactive.py         # 交互式输入
│   │
│   ├── generate_baseline.py      # 基准生成
│   ├── generate_original.py      # Original集成
│   ├── generate_flex_attention.py # FlexAttention集成
│   ├── generate_myriadlama.py    # MyriadLAMA集成
│   └── run_interactive.py        # 交互式运行器
│
├── mask_visualization.py         # Mask可视化
├── requirements.txt              # Python依赖
├── environment.yml               # Conda环境
├── environment_linux.yml         # Linux环境
│
├── GENERATE_README.md            # 生成脚本详细文档
├── README.md                     # 本文件
│
└── archived/                     # 归档文件
    ├── docs/                     # 详细文档
    ├── tests/                    # 测试文件
    ├── tools/                    # 工具脚本
    ├── analysis/                 # 分析工具
    ├── notebooks/                # Jupyter笔记本
    └── ...                       # 其他归档内容
```

## 💡 常见问题

### FlexAttention不可用

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

### CUDA内存不足

```bash
python src/generate_flex_attention.py --device cpu
```

### 数据集下载失败

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

更多问题解决：[archived/docs/DELEGATE_PROMPT.md](archived/docs/DELEGATE_PROMPT.md)

## 📚 详细文档

- **[GENERATE_README.md](GENERATE_README.md)** - 生成脚本详细说明（必读）
- **[archived/docs/](archived/docs/)** - 完整技术文档
  - [QUICKSTART.md](archived/docs/QUICKSTART.md) - 快速开始
  - [README_FLEXATTENTION.md](archived/docs/README_FLEXATTENTION.md) - FlexAttention概述
  - [ARCHITECTURE.md](archived/docs/ARCHITECTURE.md) - 架构图表
  - [实现总结.md](archived/docs/实现总结.md) - 中文实现总结

## 🔗 相关工具（已归档）

测试、分析和调试工具已移至`archived/`目录：

- **调试工具**: `archived/tools/debug_flexattention.py`
- **测试脚本**: `archived/tests/`
- **分析工具**: `archived/analysis/`
- **可视化**: `archived/plot/`
- **示例**: `archived/examples/`

这些工具仍可使用，但不是运行生成脚本的必需项。

---

**最后更新**: 2025-12-17

**状态**: ✅ 生产就绪 | 📖 已文档化 | 🧹 已整理
