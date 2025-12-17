# Archived Files - 归档文件

本目录包含了不是运行生成脚本所必需的文件，但对于深入理解、测试、调试和分析仍然有价值。

## 📁 目录说明

### 旧版本生成脚本

这些是早期版本的生成脚本，已被`src/`目录下的新版本替代：

- `generate.py` - 原始的集成生成脚本（已被`src/generate_original.py`替代）
- `baseline_generate.py` - 旧版基准生成（已被`src/generate_baseline.py`替代）
- `myriadlama_custom_attention_generate.py` - 旧版MyriadLAMA脚本（已被`src/generate_myriadlama.py`替代）

### 文档 (docs/)

完整的技术文档和指南：

- **快速开始**: `QUICKSTART.md` - 5分钟快速上手
- **FlexAttention**: `README_FLEXATTENTION.md` - FlexAttention概述
- **架构设计**: `ARCHITECTURE.md` - 可视化架构图
- **实现总结**: `实现总结.md` - 中英文实现总结
- **使用指南**: `usage/` - 详细的使用文档
- **技术细节**: `FLEX_ATTENTION_IMPLEMENTATION.md` - 技术实现细节

### 测试 (tests/)

测试脚本和验证工具：

- `flexattention/` - FlexAttention相关测试
- `myriadlama/` - MyriadLAMA数据集测试
- `unit/` - 单元测试
- `notebooks/` - Jupyter测试笔记本

### 工具 (tools/)

调试和验证工具：

- `validate_flexattention_env.py` - 环境验证
- `debug_flexattention.py` - 调试工具（详细输出）
- `example_flexattention.py` - 最小示例
- `download_resources.sh` - 资源下载脚本

### 分析 (analysis/)

结果分析工具：

- `analyze_baseline.py` - 基准分析
- `analyze_flexattention.py` - FlexAttention分析
- `flexattention_analysis.ipynb` - 交互式分析笔记本

### 可视化 (plot/)

Attention mask和结果可视化：

- `flowchart_and_attention_mask_visualization.ipynb` - 流程图和mask可视化
- `demo_visualization.py` - 演示可视化
- `test_visualization.py` - 测试可视化功能

### 示例 (examples/)

示例脚本和演示代码：

- `myriadlama_flex_example.py` - MyriadLAMA FlexAttention示例

### 笔记本 (notebooks/)

Jupyter分析和可视化笔记本：

- `flexattention_analysis.ipynb` - FlexAttention分析
- `diversity.ipynb` - 多样性分析

### 脚本 (scripts/)

批处理和辅助脚本：

- `generate_all_baselines.py` - 批量生成基准
- `start_baseline_generation.sh` - 启动基准生成

### 其他文件

- `basecopy_mcag.py` - 早期实验代码
- `position_exchange_g.py` - 位置交换实验
- `test_prompt_format.py` - 提示格式测试
- `suggestions.py` - 建议和改进想法
- `MIGRATION_GUIDE.md` - 迁移指南
- `PROMPT_UPDATE_SUMMARY.md` - 提示更新总结
- `llama_qwen_custom_attention.md` - LLaMA/Qwen自定义注意力说明

## 🔍 使用场景

### 需要深入理解项目

如果您想深入理解FlexAttention的实现细节、架构设计或测试方法，请查看：
- `docs/FLEX_ATTENTION_IMPLEMENTATION.md`
- `docs/ARCHITECTURE.md`
- `tests/`

### 需要调试或验证

如果您遇到问题需要调试，可以使用：
- `tools/debug_flexattention.py` - 详细的调试输出
- `tools/validate_flexattention_env.py` - 验证环境配置
- `tests/` - 运行测试验证功能

### 需要分析结果

如果您想分析生成结果的质量、多样性等：
- `analysis/analyze_flexattention.py` - 命令行分析
- `notebooks/flexattention_analysis.ipynb` - 交互式分析
- `plot/` - 可视化工具

### 学习和研究

如果您是研究人员想了解更多技术细节：
- `docs/` - 完整的技术文档
- `examples/` - 示例代码
- `notebooks/` - 交互式笔记本

## ⚠️ 注意事项

1. **这些文件不是运行生成脚本的必需项** - 如果您只想运行生成，请参考主README和GENERATE_README
2. **文档可能包含过时信息** - 这些文档编写于项目早期，某些细节可能已经改变
3. **测试和工具仍然可用** - 虽然归档了，但这些测试和工具仍然可以正常运行

## 🔗 返回主文档

- [主README](../README.md) - 项目主页
- [生成脚本文档](../GENERATE_README.md) - 生成脚本详细说明

---

**归档日期**: 2025-12-17
