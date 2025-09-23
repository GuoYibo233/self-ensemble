# Self-Ensemble Project - Complete Documentation

**项目名称**: Self-Ensemble  
**版本**: 2025-09-23  
**仓库**: GuoYibo233/self-ensemble  
**分支**: main  

## 📋 项目概述

这是一个专注于**自集成（Self-Ensemble）**方法的研究项目，主要用于大语言模型（LLM）的性能改进和Mixture of Experts (MoE)模型训练。项目包含完整的训练、测试、分析和部署流程。

## 🗂️ 目录结构详细说明

### 🔧 **核心配置文件**

#### **环境配置**
- **`.env`** - 环境变量配置文件，包含API密钥和路径配置
- **`environment.yml`** - Conda环境配置文件（主要环境）
- **`environment_fixed.yml`** - 修复版本的环境配置
- **`environment_simple.yml`** - 简化版环境配置，用于基本功能
- **`requirements_debug.txt`** - Python包依赖列表（调试专用）

#### **项目配置**
- **`constants.py`** - 项目常量定义，包含模型名称、路径、超参数等
- **`moe_config.py`** - MoE模型配置文件（本地环境）
- **`lab_moe_config.py`** - **实验室服务器专用MoE配置**
  - 针对Tokyo106服务器（10x RTX A6000 GPU）优化
  - 共享存储路径：`/net/tokyo100-10g/data/str01_01/yguo/`
  - 模型缓存、数据集缓存、结果输出路径配置
- **`qwen3_config.py`** - Qwen3模型专用配置文件

### 📊 **核心功能模块**

#### **数据处理**
- **`dataset.py`** - **核心数据集处理模块**
  - 支持MyriadLLaMa数据集加载
  - 数据预处理、tokenization
  - 批处理和数据加载器创建
  - 支持自定义数据格式
- **`paraphrase.py`** - 释义生成模块，用于数据增强
- **`utils.py`** - **通用工具函数库**
  - 文件I/O操作
  - 模型加载和保存
  - 评估指标计算
  - 日志记录功能

#### **模型训练与推理**
- **`generate.py`** - **主要的生成模块**
  - 支持多种LLM模型推理
  - Self-ensemble策略实现
  - 批量生成和结果聚合
- **`train_moe_lab.py`** - **实验室服务器MoE训练脚本**
  - 专为Tokyo106服务器设计
  - 支持Qwen1.5-MoE-A2.7B-Chat模型（2.7B参数，60专家）
  - 多GPU训练支持（最多10个RTX A6000）
  - 完整的训练流程：数据加载→模型初始化→训练→保存
- **`debug_generate.py`** - 调试版本的生成脚本，包含详细日志

#### **信心度评估**
- **`confidence.py`** - **模型输出信心度计算模块**
  - 实现多种信心度计算方法
  - 支持基于概率的信心度评估
  - 用于Self-ensemble中的权重分配

### 🧪 **测试与验证**

#### **单元测试**
- **`test_environment.py`** - 环境配置测试
- **`test_moe.py`** - MoE模型功能测试
- **`test_qwen3.py`** - Qwen3模型测试
- **`test_real_model.py`** - 真实模型推理测试
- **`test_quick_ensemble.py`** - Self-ensemble快速测试
- **`test_250_samples.py`** - 250样本规模测试

#### **集成测试 (`test/` 目录)**
- **`test_confidence.ipynb`** - 信心度模块测试笔记本
- **`test_dataset.ipynb`** - 数据集模块测试笔记本
- **`test_generate.ipynb`** - 生成模块测试笔记本
- **`baselines.ipynb`** - 基线模型对比测试

#### **性能测试**
- **`gpu_benchmark.py`** - GPU性能基准测试
- **`simple_myriadlama_test.py`** - MyriadLLaMa数据集简单测试
- **`quick_test_myriadlama.py`** - MyriadLLaMa快速测试脚本

### 📈 **分析与可视化 (`analysis/` 目录)**

- **`analyze_results.py`** - **结果分析主脚本**
  - 性能指标计算和统计
  - 结果对比分析
  - 生成分析报告
- **`diversity.ipynb`** - 模型输出多样性分析
- **`new_fact_occur.ipynb`** - 新事实出现频率分析
- **`num_prompts.ipynb`** - 提示数量影响分析
- **`report_accs.ipynb`** - 准确率报告生成
- **`analyze_templates.py`** - 模板分析工具
- **`moe_analysis.py`** - MoE模型专用分析工具

### 🔄 **数据下载与管理**

- **`download_moe_model.py`** - **MoE模型下载脚本**
  - 使用huggingface_hub自动下载
  - 支持Qwen1.5-MoE-A2.7B-Chat模型（28.6GB）
  - 自动保存到共享存储位置
  - 支持断点续传
- **`download_dataset.py`** - **数据集下载脚本**
  - 下载MyriadLLaMa数据集到共享存储
  - 备用方案：创建测试数据集（100训练+20验证样本）
  - 自动保存到`/net/tokyo100-10g/data/str01_01/yguo/datasets/`

### 🚀 **部署与自动化 (`scripts/` 目录)**

- **`deploy_to_lab_server.sh`** - 实验室服务器部署脚本
- **`main.sh`** - 主要实验运行脚本
- **`ensemble.sh`** - Self-ensemble实验脚本
- **`diversity.sh`** - 多样性分析脚本
- **`lemmaize_ensemble.sh`** - 词形还原ensemble脚本
- **`run_moe_experiments.sh`** - MoE实验批处理脚本

### 📚 **文档与指南**

#### **部署指南**
- **`LAB_DEPLOYMENT_GUIDE.md`** - **实验室服务器部署完整指南**
  - Tokyo106服务器配置详情
  - 环境设置步骤
  - 常见问题解决方案
- **`MIGRATION_GUIDE.md`** - 环境迁移指南
- **`QUICK_MIGRATION.md`** - 快速迁移步骤
- **`setup_environment.md`** - 环境设置详细说明
- **`vscode_debug_guide.md`** - VS Code调试配置指南
- **`test_guide.md`** - 测试运行指南

#### **启动脚本**
- **`setup_once.ps1`** - Windows一次性环境设置脚本
- **`start_debug.ps1`** - Windows调试启动脚本
- **`start_debug.bat`** - Windows批处理启动脚本

### 🎯 **比较分析**

- **`myriadlama_comparison.py`** - **与MyriadLLaMa基线对比**
  - 性能基准对比
  - 多维度评估指标
  - 结果可视化

### 📊 **实验结果与数据**

- **`quick_self_ensemble_test_20250915_054752.json`** - 快速自集成测试结果
- **`real_model_test_20250915_051636.json`** - 真实模型测试结果
- **`real_model_test_20250915_052515.json`** - 真实模型测试结果（第二次）
- **`debug_quickstart.html`** - 调试快速开始网页版
- **`debug_quickstart.md`** - 调试快速开始Markdown版

### 👀 **结果查看**

- **`view_results.py`** - 实验结果查看器
- **`view_prompt_answer.py`** - 提示和回答查看工具

### 📁 **数据目录**

- **`datasets/`** - 本地数据集存储
- **`mock_data/`** - 模拟测试数据
- **`__pycache__/`** - Python字节码缓存

### ⚙️ **监控工具**

- **`monitor_moe_training.py`** - **MoE训练过程监控**
  - 实时训练进度跟踪
  - GPU使用率监控
  - 损失函数可视化
  - 早停机制支持

## 🔗 **核心工作流程**

### **1. 环境设置流程**
1. 运行 `setup_once.ps1` 进行初始环境设置
2. 使用 `environment.yml` 创建conda环境
3. 配置 `.env` 文件中的路径和API密钥

### **2. 实验室部署流程**
1. 参考 `LAB_DEPLOYMENT_GUIDE.md` 进行服务器配置
2. 运行 `download_moe_model.py` 下载模型（28.6GB）
3. 运行 `download_dataset.py` 准备数据集
4. 使用 `train_moe_lab.py` 开始MoE训练

### **3. 标准实验流程**
1. 数据准备：`dataset.py` → 数据加载和预处理
2. 模型训练：`train_moe_lab.py` → MoE模型训练
3. 推理生成：`generate.py` → Self-ensemble推理
4. 结果分析：`analysis/analyze_results.py` → 性能分析
5. 结果可视化：使用`analysis/`目录下的Jupyter notebooks

### **4. 测试验证流程**
1. 环境测试：`test_environment.py`
2. 模块测试：运行`test/`目录下的notebooks
3. 性能测试：`gpu_benchmark.py`
4. 集成测试：`test_quick_ensemble.py`

## 🛠️ **技术架构**

### **模型支持**
- **MoE模型**: Qwen1.5-MoE-A2.7B-Chat (2.7B参数, 60专家)
- **标准LLM**: Qwen系列, LLaMa系列
- **训练框架**: PyTorch 2.5.1, Transformers 4.56.2

### **硬件要求**
- **实验室环境**: Tokyo106服务器
  - 10x NVIDIA RTX A6000 (48GB each)
  - 251GB RAM, 64 CPU cores
  - 65TB 共享存储空间
- **最小需求**: 1x GPU (>16GB), 32GB RAM

### **存储架构**
```
共享存储: /net/tokyo100-10g/data/str01_01/yguo/
├── models/moe/                    # MoE模型存储 (28.6GB)
├── datasets/myriadlama/           # 数据集存储
├── experiments/self-ensemble/     # 实验结果
└── cache/huggingface/            # Hugging Face缓存
```

## 📝 **使用说明**

### **新用户快速开始**
1. 阅读 `QUICK_MIGRATION.md`
2. 运行 `setup_once.ps1` 配置环境
3. 使用 `test_environment.py` 验证配置
4. 运行 `simple_myriadlama_test.py` 进行基础测试

### **实验室用户**
1. 参考 `LAB_DEPLOYMENT_GUIDE.md` 完成服务器配置
2. 确保模型和数据集已下载完成
3. 运行 `train_moe_lab.py` 开始训练
4. 使用 `monitor_moe_training.py` 监控进度

### **开发者**
1. 参考 `vscode_debug_guide.md` 配置调试环境
2. 运行单元测试确保代码质量
3. 使用 `analysis/` 目录进行结果分析
4. 参考 `MIGRATION_GUIDE.md` 了解架构变更

## 🔄 **版本历史**

- **2025-09-23**: 完成实验室服务器部署和MoE模型集成
- **2025-09-15**: 添加Self-ensemble测试和结果分析
- **初始版本**: 基础Self-ensemble框架实现

## 🚨 **重要提示**

1. **模型下载**: MoE模型文件很大（28.6GB），首次下载需要数小时
2. **GPU内存**: MoE训练至少需要16GB GPU内存
3. **共享存储**: 实验室环境必须使用共享存储路径
4. **环境隔离**: 每个用户应使用独立的conda环境
5. **资源监控**: 训练前检查GPU可用性，避免资源冲突

## 📧 **联系信息**

- **项目维护者**: GuoYibo233
- **实验室环境**: Tokyo106服务器
- **技术支持**: 参考相关文档或联系项目团队

---
*文档最后更新: 2025-09-23*  
*生成工具: GitHub Copilot Assistant*