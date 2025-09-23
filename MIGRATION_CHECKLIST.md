# 环境迁移检查清单 - Quick Migration Checklist

**迁移日期**: _____________  
**目标环境**: _____________  
**迁移执行者**: _____________  

## ✅ 迁移前准备

### 📋 **硬件要求确认**
- [ ] GPU: 至少1个GPU (推荐RTX A6000 48GB 或同等级)
- [ ] 内存: 至少32GB RAM (推荐64GB+)
- [ ] 存储: 至少100GB可用空间 (模型28.6GB + 数据集 + 结果)
- [ ] 网络: 稳定的互联网连接 (用于模型下载)

### 🔧 **软件环境确认**
- [ ] 操作系统: Linux (推荐) 或 Windows with WSL
- [ ] Python 3.9+
- [ ] CUDA 12.x 支持
- [ ] Conda 或 Miniconda 已安装
- [ ] Git 已安装

## 📁 **文件迁移清单**

### 🔥 **必需文件 (Phase 1)**
```bash
# 核心训练文件
train_moe_lab.py           ✅ 复制完成 □
lab_moe_config.py          ✅ 复制完成 □ ⚠️ 需修改路径
download_moe_model.py      ✅ 复制完成 □
download_dataset.py        ✅ 复制完成 □

# 核心模块
dataset.py                 ✅ 复制完成 □
utils.py                   ✅ 复制完成 □
constants.py               ✅ 复制完成 □
generate.py                ✅ 复制完成 □

# 环境配置
environment.yml            ✅ 复制完成 □
.env                       ✅ 复制完成 □ ⚠️ 需配置密钥
```

### 📚 **文档文件 (Phase 2)**
```bash
PROJECT_DOCUMENTATION.md   ✅ 复制完成 □
FILE_INVENTORY.md          ✅ 复制完成 □
LAB_DEPLOYMENT_GUIDE.md    ✅ 复制完成 □
MIGRATION_CHECKLIST.md     ✅ 复制完成 □
```

### 🧪 **测试文件 (Phase 3)**
```bash
test_environment.py        ✅ 复制完成 □
simple_myriadlama_test.py  ✅ 复制完成 □
test_moe.py               ✅ 复制完成 □
```

### 📊 **分析工具 (Optional)**
```bash
analysis/                  ✅ 复制完成 □
monitor_moe_training.py    ✅ 复制完成 □
confidence.py              ✅ 复制完成 □
```

## 🔧 **环境配置步骤**

### Step 1: 创建Conda环境
```bash
# 1.1 创建环境
conda env create -f environment.yml
□ 执行完成

# 1.2 激活环境
conda activate gyb-self-ensemble
□ 执行完成

# 1.3 验证PyTorch和CUDA
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
预期输出: 
- PyTorch 2.5.1+cu121
- True
□ 验证通过
```

### Step 2: 配置路径
```bash
# 2.1 修改 lab_moe_config.py 中的路径
# 将路径修改为新环境的存储位置
□ 路径修改完成

# 2.2 配置 .env 文件
# 添加必要的API密钥和环境变量
□ 环境变量配置完成
```

### Step 3: 目录结构创建
```bash
# 3.1 创建必要目录
mkdir -p ~/shared_storage/models/moe
mkdir -p ~/shared_storage/datasets/myriadlama
mkdir -p ~/shared_storage/experiments/self-ensemble
□ 目录创建完成

# 3.2 更新配置文件中的路径指向
□ 路径更新完成
```

## 🧪 **环境验证步骤**

### Test 1: 基础环境测试
```bash
python test_environment.py
预期输出: 所有检查通过
□ 测试通过 ⚠️ 有问题: ________________
```

### Test 2: GPU可用性测试
```bash
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
预期输出: GPU数量 > 0
□ 测试通过 ⚠️ 有问题: ________________
```

### Test 3: 依赖包测试
```bash
python -c "from transformers import AutoConfig; print('Transformers OK')"
python -c "from huggingface_hub import snapshot_download; print('HuggingFace Hub OK')"
□ 测试通过 ⚠️ 有问题: ________________
```

## 📥 **数据下载步骤**

### Step 1: 下载MoE模型 (约28.6GB)
```bash
python download_moe_model.py
预期时间: 1-3小时 (取决于网络速度)
□ 下载完成 ⚠️ 下载失败: ________________
```

### Step 2: 准备数据集
```bash
python download_dataset.py
预期输出: 测试数据集创建成功 (100训练 + 20验证样本)
□ 数据集准备完成
```

### Step 3: 验证下载结果
```bash
# 3.1 检查模型文件
ls ~/shared_storage/models/moe/qwen1.5_moe_a2.7b_chat/
预期: 看到8个model-*.safetensors文件 + config.json等
□ 模型文件齐全

# 3.2 检查数据集文件
ls ~/shared_storage/datasets/myriadlama/test_data/
预期: train.jsonl, validation.jsonl, dataset_info.json
□ 数据集文件齐全
```

## 🚀 **功能测试步骤**

### Test 1: 简单测试
```bash
python simple_myriadlama_test.py
预期输出: 基础功能正常
□ 测试通过
```

### Test 2: MoE模型加载测试
```bash
python -c "
from lab_moe_config import LAB_SERVER_CONFIG
from transformers import AutoConfig
config = AutoConfig.from_pretrained(LAB_SERVER_CONFIG['model_cache_dir'] + '/qwen1.5_moe_a2.7b_chat/')
print(f'模型类型: {config.model_type}')
print(f'专家数量: {config.num_experts}')
"
预期输出: 
- 模型类型: qwen2_moe
- 专家数量: 60
□ 模型加载测试通过
```

### Test 3: 最小训练测试
```bash
python train_moe_lab.py --model_name qwen1.5_moe_a2.7b_chat --max_samples 2 --batch_size 1 --max_steps 1 --experiment_name migration_test --output_dir ~/data/results
预期: 训练开始且无错误
□ 最小训练测试通过
```

## ⚠️ **常见问题解决方案**

### 问题1: CUDA不可用
```bash
# 检查NVIDIA驱动
nvidia-smi

# 重新安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
□ 解决方案已尝试
```

### 问题2: 模型下载失败
```bash
# 检查网络连接
ping huggingface.co

# 设置代理 (如果需要)
export HTTP_PROXY=your_proxy
export HTTPS_PROXY=your_proxy
□ 解决方案已尝试
```

### 问题3: 内存不足
```bash
# 减少批次大小和样本数量
python train_moe_lab.py --batch_size 1 --max_samples 1
□ 解决方案已尝试
```

## 🎉 **迁移完成确认**

### 最终验证清单
- [ ] 环境创建成功并可激活
- [ ] 所有依赖包正确安装
- [ ] GPU可用且CUDA工作正常
- [ ] MoE模型下载完成 (28.6GB)
- [ ] 数据集准备完成
- [ ] 基础功能测试通过
- [ ] 最小训练测试成功
- [ ] 配置文件路径正确
- [ ] 文档和清单已更新

### 性能基准测试 (可选)
- [ ] GPU基准测试: `python gpu_benchmark.py`
- [ ] 完整训练测试: 10样本，5步训练
- [ ] 结果分析测试: 使用analysis工具

## 📝 **迁移总结**

**迁移状态**: □ 成功 □ 部分成功 □ 失败  
**遇到的主要问题**: ________________________________  
**解决方案**: _____________________________________  
**建议改进**: _____________________________________  

**下一步行动**:
- [ ] 进行更大规模的训练实验
- [ ] 测试Self-Ensemble功能
- [ ] 运行完整的分析流程

---
**迁移完成时间**: _______________  
**验证人员**: _______________  

*此检查清单确保环境迁移的完整性和可用性。建议保存此文件作为迁移记录。*