# Recent Updates Summary

**更新日期 / Update Date**: 2025-10-20

## 本次更新内容 / Updates in This Session

### 1. 🎯 新增Origin基线方法 / Added Origin Baseline Method

**目的**: 提供真正的baseline用于评估paraphrase和ensemble方法的效果

**新功能**:
- 在`generate.py`中添加`--method origin`选项
- 只使用原始问题，不使用任何paraphrase
- 自动进行lemmatization
- 输出文件: `origin.feather`

**使用方法**:
```bash
python generate.py --method origin --dataset webqa --model llama3.2_3b_it
```

**输出位置**:
```
/net/tokyo100-10g/data/str01_01/y-guo/datasets/webqa/llama3.2_3b_it/origin.feather
```

---

### 2. 🔧 统一所有路径配置 / Unified All Path Configurations

**目的**: 从原作者路径迁移到用户自己的路径，使用HuggingFace Hub集中管理

#### 改动详情 / Changes

**A. 模型路径 (constants.py)**
- 设置`HF_HOME = /net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache`
- 所有模型改为HuggingFace Hub ID格式:
  - `llama3.2_3b_it`: `meta-llama/Llama-3.2-3B-Instruct`
  - `llama3.1_8b_it`: `meta-llama/Llama-3.1-8B-Instruct`
  - `qwen2.5_7b`: `Qwen/Qwen2.5-7B`
  - 等等...

**B. 数据集路径 (dataset.py)**
- `DATATASET_ROOT = /net/tokyo100-10g/data/str01_01/y-guo/datasets`

**C. 其他文件**
- `.env`: 更新PYTHONPATH
- `test/test_generate.ipynb`: 更新路径
- `test/test_confidence.ipynb`: 更新路径

#### 优势 / Benefits
- ✅ 完全独立的工作环境
- ✅ 自动模型下载和缓存
- ✅ 统一目录结构
- ✅ 无需软链接或手动下载

#### 首次使用需要 / First Time Setup Required
```bash
# 1. 登录HuggingFace
hf auth login

# 2. 访问以下链接接受LLaMA许可协议
# https://huggingface.co/meta-llama/Llama-3.2-1B
# https://huggingface.co/meta-llama/Llama-3.1-8B
```

---

## 更新的文档 / Updated Documentation

以下文档已更新以反映最新改动:

1. **README.md**
   - 更新方法对比表格，添加`origin`方法
   - 更新使用示例，展示所有方法

2. **FLEXATTENTION_USAGE.md**
   - 添加方法概览表格
   - 添加详细使用示例
   - 说明各方法适用场景

3. **docs/QUICK_REFERENCE.md**
   - 添加`origin`方法说明
   - 更新方法对比表格

4. **CHANGELOG.md**
   - 添加两个新条目:
     - "Unified Model Paths and Dataset Paths"
     - "Add Origin Baseline Method"

5. **RECENT_UPDATES.md** (本文件)
   - 新建，总结最近的所有更新

---

## 目录结构 / Directory Structure

更新后的目录结构：

```
/net/tokyo100-10g/data/str01_01/y-guo/
├── huggingface_cache/           # HuggingFace模型缓存
│   └── hub/
│       ├── models--meta-llama--Llama-3.2-3B-Instruct/
│       ├── models--meta-llama--Llama-3.1-8B-Instruct/
│       └── models--Qwen--Qwen2.5-7B/
│
└── datasets/                    # 数据集和生成结果
    ├── webqa/
    │   └── llama3.2_3b_it/
    │       ├── paraphrases_dataset/      # paraphrase数据
    │       ├── origin.feather            # 新增: origin baseline
    │       ├── per_prompt.feather
    │       ├── ensemble_avg-6.feather
    │       ├── ensemble_max-6.feather
    │       └── flex_attention-5.feather
    │
    └── myriadlama/
        └── {model_name}/
            └── ...
```

---

## 所有可用的生成方法 / All Available Generation Methods

| Method | Description | Command | Output File |
|--------|-------------|---------|-------------|
| **origin** | 原始问题baseline | `python generate.py --method origin` | `origin.feather` |
| **per_prompt** | 每个paraphrase单独生成 | `python generate.py --method per_prompt` | `per_prompt.feather` |
| **avg** | Logit平均融合 | `python generate.py --method avg --num_ensemble 6` | `ensemble_avg-6.feather` |
| **max** | Logit最大值融合 | `python generate.py --method max --num_ensemble 6` | `ensemble_max-6.feather` |
| **weighted_avg** | 加权平均融合 | `python generate.py --method weighted_avg --num_ensemble 6` | `ensemble_weighted_avg-6.feather` |
| **weighted_max** | 加权最大值融合 | `python generate.py --method weighted_max --num_ensemble 6` | `ensemble_weighted_max-6.feather` |
| **flex_attention** | Attention层融合 | `python flex_attention_generate.py --num_paraphrases 5` | `flex_attention-5.feather` |

---

## 性能对比 / Performance Comparison

| Method | Forward Passes | Efficiency | Fusion Level |
|--------|----------------|------------|--------------|
| origin | 1× per sample | Fastest | None |
| per_prompt | 6× per sample | Baseline | None |
| avg/max | 6× per sample | Same as per_prompt | Logit |
| weighted_* | 6× per sample | Same as per_prompt | Logit + confidence |
| **flex_attention** | **1× per sample** | **Most efficient** | **Attention** |

---

## 快速开始 / Quick Start

### 1. 首次设置 / First Time Setup

```bash
# 登录HuggingFace
hf auth login

# 接受LLaMA许可协议（访问网页）
# https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

### 2. 生成baseline / Generate Baseline

```bash
cd /home/y-guo/self-ensemble/self-ensemble

# 生成origin baseline
python generate.py \
    --method origin \
    --dataset webqa \
    --model llama3.2_3b_it
```

### 3. 生成其他方法 / Generate Other Methods

```bash
# Per-prompt
python generate.py --method per_prompt --dataset webqa --model llama3.2_3b_it

# Ensemble
python generate.py --method avg --dataset webqa --model llama3.2_3b_it --num_ensemble 6

# FlexAttention
python flex_attention_generate.py --dataset webqa --model llama3.2_3b_it --num_paraphrases 5
```

### 4. 分析结果 / Analyze Results

```bash
# 使用analysis工具对比所有方法
python analysis/analyze_flexattention.py --dataset webqa --model llama3.2_3b_it
```

---

## 注意事项 / Important Notes

1. **首次下载模型需要时间**
   - LLaMA 3.2 3B: ~15-30分钟
   - LLaMA 3.1 8B: ~30-60分钟
   - 取决于网络速度

2. **确保有足够磁盘空间**
   - 每个模型: 6-30GB
   - 数据集和结果: 额外10-20GB

3. **HuggingFace认证**
   - 需要先登录: `hf auth login`
   - 需要接受LLaMA模型许可协议

4. **路径权限**
   - 确保对`/net/tokyo100-10g/data/str01_01/y-guo/`有写权限

---

## 问题排查 / Troubleshooting

### 问题1: 401 Unauthorized错误
```bash
# 解决方案: 登录HuggingFace
hf auth login
```

### 问题2: 模型下载慢
```bash
# 可选: 使用镜像（如果在国内）
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题3: 磁盘空间不足
```bash
# 检查可用空间
df -h /net/tokyo100-10g/data/str01_01/y-guo/

# 如果空间不足，可以删除旧的输出文件
rm -rf /net/tokyo100-10g/data/str01_01/y-guo/datasets/webqa/old_model/
```

---

## 下一步 / Next Steps

1. ✅ 完成路径迁移
2. ✅ 添加origin baseline方法
3. ✅ 更新所有文档
4. ⏳ 测试生成200个样本
5. ⏳ 对比所有方法的性能
6. ⏳ 分析结果并生成报告

---

**最后更新 / Last Updated**: 2025-10-20
**作者 / Author**: GitHub Copilot
**状态 / Status**: ✅ 完成 / Completed
