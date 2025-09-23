# Self-Ensemble Project - File Inventory

**生成时间**: 2025-09-23  
**项目状态**: 实验室服务器部署完成，MoE训练环境就绪  

## 📁 文件分类清单

### 🔥 **最重要的核心文件**
| 文件                    | 用途                  | 优先级 |
| ----------------------- | --------------------- | ------ |
| `train_moe_lab.py`      | **MoE模型训练主脚本** | ⭐⭐⭐⭐⭐  |
| `lab_moe_config.py`     | **实验室服务器配置**  | ⭐⭐⭐⭐⭐  |
| `download_moe_model.py` | **MoE模型下载**       | ⭐⭐⭐⭐⭐  |
| `download_dataset.py`   | **数据集下载**        | ⭐⭐⭐⭐⭐  |
| `dataset.py`            | **数据处理核心模块**  | ⭐⭐⭐⭐   |
| `generate.py`           | **推理生成主脚本**    | ⭐⭐⭐⭐   |
| `utils.py`              | **通用工具函数**      | ⭐⭐⭐⭐   |

### 🛠️ **配置文件**
| 文件                | 用途                 | 状态     |
| ------------------- | -------------------- | -------- |
| `environment.yml`   | Conda环境配置        | ✅ 已验证 |
| `lab_moe_config.py` | 实验室服务器路径配置 | ✅ 已配置 |
| `moe_config.py`     | 本地MoE配置          | 📝 备用   |
| `constants.py`      | 项目常量定义         | ✅ 稳定   |
| `.env`              | 环境变量配置         | ⚠️ 需配置 |

### 🧪 **测试文件**
| 文件                        | 测试对象          | 状态     |
| --------------------------- | ----------------- | -------- |
| `test_environment.py`       | 环境配置          | ✅ 通过   |
| `test_moe.py`               | MoE模型功能       | 📝 待测试 |
| `test_quick_ensemble.py`    | Self-ensemble功能 | 📝 待测试 |
| `simple_myriadlama_test.py` | 基础数据集测试    | ✅ 可用   |

### 📊 **分析工具**
| 文件                      | 分析内容       | 位置        |
| ------------------------- | -------------- | ----------- |
| `analyze_results.py`      | 结果分析主脚本 | `analysis/` |
| `diversity.ipynb`         | 输出多样性分析 | `analysis/` |
| `monitor_moe_training.py` | 训练过程监控   | 根目录      |
| `moe_analysis.py`         | MoE专用分析    | 根目录      |

### 🚀 **部署脚本**
| 文件                      | 部署目标     | 状态     |
| ------------------------- | ------------ | -------- |
| `deploy_to_lab_server.sh` | 实验室服务器 | ✅ 已完成 |
| `setup_once.ps1`          | Windows环境  | 📝 通用   |
| `run_moe_experiments.sh`  | MoE批量实验  | 📝 待测试 |

### 📚 **文档文件**
| 文件                       | 内容               | 重要性 |
| -------------------------- | ------------------ | ------ |
| `LAB_DEPLOYMENT_GUIDE.md`  | **实验室部署指南** | ⭐⭐⭐⭐⭐  |
| `PROJECT_DOCUMENTATION.md` | **项目完整文档**   | ⭐⭐⭐⭐⭐  |
| `MIGRATION_GUIDE.md`       | 环境迁移指南       | ⭐⭐⭐    |
| `vscode_debug_guide.md`    | VS Code调试配置    | ⭐⭐     |

## 🎯 **关键工作流程文件**

### **MoE训练完整流程**
```
1. lab_moe_config.py        (配置路径)
2. download_moe_model.py    (下载模型 28.6GB)
3. download_dataset.py      (准备数据集)
4. train_moe_lab.py         (开始训练)
5. monitor_moe_training.py  (监控进度)
6. analyze_results.py       (分析结果)
```

### **Self-Ensemble实验流程**
```
1. dataset.py              (数据预处理)
2. generate.py              (批量推理)
3. confidence.py            (信心度计算)
4. analysis/diversity.ipynb (多样性分析)
```

## 🔧 **环境迁移清单**

### **必需文件 (新环境必须复制)**
- [x] `train_moe_lab.py` - 训练脚本
- [x] `lab_moe_config.py` - 服务器配置
- [x] `download_moe_model.py` - 模型下载
- [x] `download_dataset.py` - 数据下载
- [x] `environment.yml` - 环境配置
- [x] `dataset.py` - 数据处理
- [x] `utils.py` - 工具函数
- [x] `constants.py` - 常量定义

### **配置文件 (需要修改路径)**
- [ ] `lab_moe_config.py` - 更新服务器路径
- [ ] `.env` - 配置API密钥和环境变量

### **可选文件 (根据需要)**
- `generate.py` - 推理生成
- `confidence.py` - 信心度计算  
- `analysis/` - 分析工具
- `test/` - 测试套件

## 🚨 **当前状态总结**

### ✅ **已完成**
1. **实验室环境配置** - Tokyo106服务器已就绪
2. **MoE模型下载** - Qwen1.5-MoE-A2.7B-Chat (28.6GB) 已下载完成
3. **数据集准备** - 测试数据集已创建
4. **训练环境验证** - PyTorch 2.5.1 + 10x RTX A6000 已确认
5. **基础功能测试** - 模型加载、配置文件均正常

### 🔄 **进行中**
1. **MoE训练优化** - 多GPU并行训练参数调优
2. **性能基准测试** - 与基线模型对比实验
3. **结果分析工具** - 自动化分析流程完善

### 📋 **待完成**
1. **大规模训练实验** - 使用完整数据集进行训练
2. **Self-Ensemble集成** - 与MoE模型结合测试
3. **生产环境部署** - 稳定版本打包和部署

## 📞 **迁移支持**

如果其他agent需要接手这个项目：
1. 首先阅读 `PROJECT_DOCUMENTATION.md` 了解全貌
2. 参考 `LAB_DEPLOYMENT_GUIDE.md` 配置环境
3. 运行 `test_environment.py` 验证环境
4. 从简单的 `simple_myriadlama_test.py` 开始测试

---
*此清单涵盖了项目的所有关键信息，建议新的agent先从核心文件开始理解。*