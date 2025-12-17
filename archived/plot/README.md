# FlexAttention 可视化脚本 / FlexAttention Visualization Scripts

本目录包含用于可视化 FlexAttention 代码流程和注意力掩码的脚本。
This directory contains scripts for visualizing FlexAttention code flowchart and attention masks.

---

## 📊 主要文件 / Main Files

### 1. `flowchart_and_attention_mask_visualization.ipynb`

**完整的 Jupyter 笔记本，提供交互式可视化**
**Complete Jupyter notebook with interactive visualizations**

#### 功能特性 / Features:

- ✅ **代码流程图** - 展示从输入到输出的完整处理流程
  - **Code flowchart** - Shows complete processing pipeline from input to output

- ✅ **注意力掩码可视化** - 精确展示不同场景下的掩码矩阵形状
  - **Attention mask visualization** - Precisely shows mask matrix shapes in different scenarios

- ✅ **可交互参数** - 所有配置集中在配置字典中，方便修改和实验
  - **Interactive parameters** - All configs centralized for easy modification and experimentation

- ✅ **智能采样** - 处理大型序列时保持结构可见性
  - **Smart sampling** - Maintains structure visibility for large sequences

- ✅ **注意力模式分析** - 对比编码和生成阶段的注意力行为
  - **Attention pattern analysis** - Compares encoding vs generation phase behavior

- ✅ **导出功能** - 保存所有可视化为高分辨率图片
  - **Export functionality** - Save all visualizations as high-resolution images

#### 使用方法 / Usage:

```bash
# 1. 安装依赖 / Install dependencies
pip install jupyter matplotlib numpy

# 2. 启动 Jupyter / Start Jupyter
jupyter notebook flowchart_and_attention_mask_visualization.ipynb

# 3. 运行所有单元格查看预设示例 / Run all cells to see preset examples
# 4. 修改配置字典进行自定义实验 / Modify config dictionaries for custom experiments
```

#### 三种预设场景 / Three Preset Scenarios:

1. **小型示例 (Small Example)** - 3个改写，每个15令牌，完整展示
   - 3 paraphrases, 15 tokens each, full display
   - 适合理解基本原理 / Good for understanding basic principles

2. **中型示例 (Medium Example)** - 5个改写，每个25令牌，智能采样
   - 5 paraphrases, 25 tokens each, smart sampling
   - 展示采样策略 / Shows sampling strategy

3. **大型示例 (Large Example)** - 5个改写，每个50令牌，真实场景
   - 5 paraphrases, 50 tokens each, realistic scenario
   - 模拟真实使用 / Simulates real usage

#### 自定义配置示例 / Custom Configuration Example:

```python
CUSTOM_CONFIG = {
    'num_paraphrases': 4,              # 改写数量 / Number of paraphrases
    'tokens_per_paraphrase': 30,       # 每个改写的令牌数 / Tokens per paraphrase
    'separator_tokens': 3,             # 分隔符令牌数 / Separator tokens
    'num_generated_tokens': 10,        # 生成的令牌数 / Generated tokens
    'display_mode': 'sampled',         # 'full' 或 'sampled' / 'full' or 'sampled'
    'max_display_positions': 30,       # 最大显示位置数 / Max display positions
}
```

---

### 2. `test_visualization.py`

**独立测试脚本，验证可视化函数的正确性**
**Standalone test script to verify visualization functions**

#### 用途 / Purpose:

- 验证注意力掩码函数的正确性 / Verify attention mask function correctness
- 测试关键属性（因果约束、分段隔离、融合关注）/ Test key properties (causal, isolation, fusion)
- 无需 Jupyter 即可运行 / Runs without Jupyter

#### 运行方法 / How to Run:

```bash
# 安装依赖 / Install dependencies
pip install numpy

# 运行测试 / Run tests
python3 test_visualization.py
```

#### 预期输出 / Expected Output:

```
✓ 因果约束违规 / Causal violations: 0 (should be 0)
✓ 编码阶段跨段关注 / Encoding cross-segment attention: 0 (should be 0)
✓ 第一个生成令牌可关注位置 / First generated token attends to: 50/50
✓ 所有测试通过！/ All tests passed!
```

---

## 🎨 可视化示例 / Visualization Examples

### 代码流程图 / Code Flowchart

流程图展示了 FlexAttention 的完整处理流程：
The flowchart shows the complete FlexAttention processing pipeline:

1. **输入准备** - 加载问题和改写 / **Input preparation** - Load question and paraphrases
2. **拼接处理** - 拼接改写并追踪位置 / **Concatenation** - Concatenate with position tracking
3. **创建掩码** - 生成 FlexAttention 掩码函数 / **Create mask** - Generate FlexAttention mask function
4. **模型打补丁** - 使用 FlexAttention 替换注意力层 / **Patch model** - Replace attention layers with FlexAttention
5. **编码阶段** - 分段隔离注意力 / **Encoding phase** - Segment-isolated attention
6. **生成循环** - 自回归生成，融合关注 / **Generation loop** - Auto-regressive generation with fusion
7. **解码输出** - 返回生成的文本 / **Decode output** - Return generated text

### 注意力掩码矩阵 / Attention Mask Matrix

掩码矩阵展示了注意力模式：
The mask matrix shows attention patterns:

- **绿色方块 (■)** = 可以关注 / Can attend
- **白色方块 (·)** = 不可关注 / Cannot attend
- **红色虚线** = 分段边界 / Segment boundary
- **蓝色虚线** = 生成开始 / Generation start

#### 编码阶段模式 / Encoding Phase Pattern:
```
         Seg1  Seg2  Seg3  Seg4  Seg5
  Seg1   ■■    ··    ··    ··    ··     (Isolated)
  Seg2   ··    ■■    ··    ··    ··     (Isolated)
  Seg3   ··    ··    ■■    ··    ··     (Isolated)
  Seg4   ··    ··    ··    ■■    ··     (Isolated)
  Seg5   ··    ··    ··    ··    ■■     (Isolated)
```

#### 生成阶段模式 / Generation Phase Pattern:
```
         Seg1  Seg2  Seg3  Seg4  Seg5  Gen
  Gen1   ■■    ■■    ■■    ■■    ■■    ■   (Fusion)
  Gen2   ■■    ■■    ■■    ■■    ■■    ■■  (Fusion)
```

---

## 📝 关键概念 / Key Concepts

### 分段隔离 / Segment Isolation

在编码阶段，每个改写（paraphrase）在其自己的分段内独立处理，互不干扰：
During encoding, each paraphrase is processed independently within its segment:

- **目的** - 保持每个改写的独立性 / **Purpose** - Maintain independence of each paraphrase
- **实现** - 掩码函数只允许同一分段内的令牌相互关注 / **Implementation** - Mask only allows within-segment attention
- **效果** - 防止不同改写之间的信息泄露 / **Effect** - Prevents information leakage between paraphrases

### 融合生成 / Fusion Generation

在生成阶段，新生成的令牌可以关注所有之前的内容（所有改写）：
During generation, newly generated tokens can attend to all previous content (all paraphrases):

- **目的** - 融合来自多个改写的信息 / **Purpose** - Fuse information from multiple paraphrases
- **实现** - 生成令牌的掩码函数允许关注所有位置 / **Implementation** - Generated tokens' mask allows attending to all positions
- **效果** - 生成更准确、更鲁棒的答案 / **Effect** - Generate more accurate and robust answers

### 因果约束 / Causal Constraint

所有注意力都必须遵守因果约束（不能关注未来）：
All attention must respect the causal constraint (cannot attend to future):

- **规则** - `q_idx >= kv_idx` / **Rule** - `q_idx >= kv_idx`
- **原因** - 保持自回归生成的正确性 / **Reason** - Maintain auto-regressive generation correctness
- **验证** - 测试脚本会验证此约束 / **Verification** - Test script verifies this constraint

---

## 🔧 修改和扩展 / Modification and Extension

### 修改流程图外观 / Modify Flowchart Appearance

在笔记本中修改 `FLOWCHART_CONFIG` 字典：
Modify the `FLOWCHART_CONFIG` dictionary in the notebook:

```python
FLOWCHART_CONFIG = {
    'figure_size': (14, 16),          # 调整图表大小 / Adjust figure size
    'box_width': 3.5,                 # 调整框宽度 / Adjust box width
    'box_height': 0.6,                # 调整框高度 / Adjust box height
    'vertical_spacing': 1.2,          # 调整垂直间距 / Adjust spacing
    
    # 修改颜色方案 / Modify color scheme
    'color_input': '#E3F2FD',         
    'color_processing': '#FFF3E0',    
    'color_attention': '#F3E5F5',     
    'color_generation': '#E8F5E9',    
    'color_output': '#FCE4EC',        
}
```

### 添加新的测试场景 / Add New Test Scenarios

创建新的配置字典：
Create a new configuration dictionary:

```python
MY_CUSTOM_CONFIG = {
    'num_paraphrases': 7,              # 更多改写 / More paraphrases
    'tokens_per_paraphrase': 40,       # 更长的改写 / Longer paraphrases
    'separator_tokens': 5,             # 更长的分隔符 / Longer separator
    'num_generated_tokens': 20,        # 更多生成令牌 / More generated tokens
    'display_mode': 'sampled',         
    'max_display_positions': 40,       # 显示更多位置 / Show more positions
}

# 使用新配置 / Use new config
fig, matrix, positions = visualize_attention_mask(MY_CUSTOM_CONFIG)
```

---

## 📚 相关文档 / Related Documentation

- **实现细节** - `../flex_attention_generate.py` - FlexAttention 实现代码
  - **Implementation** - FlexAttention implementation code

- **架构说明** - `../docs/ARCHITECTURE.md` - 架构图和说明
  - **Architecture** - Architecture diagrams and explanations

- **测试示例** - `../test_mask_visualization.py` - 更多测试示例
  - **Test examples** - More test examples

- **使用指南** - `../FLEXATTENTION_USAGE.md` - 使用说明
  - **Usage guide** - Usage instructions

---

## ❓ 常见问题 / FAQ

### Q: 如何在没有 Jupyter 的情况下查看可视化？
### Q: How to view visualizations without Jupyter?

**A:** 运行笔记本后，使用导出功能保存图片：
**A:** After running the notebook, use export functionality to save images:

```python
# 在笔记本的最后一个单元格运行 / Run in the last cell of notebook
# 图片会保存到 attention_mask_outputs/ 目录
# Images will be saved to attention_mask_outputs/ directory
```

### Q: 如何验证掩码函数的正确性？
### Q: How to verify mask function correctness?

**A:** 运行测试脚本：
**A:** Run the test script:

```bash
python3 test_visualization.py
```

### Q: 可以用于其他模型吗？
### Q: Can this be used for other models?

**A:** 是的！只需修改配置参数来匹配你的模型：
**A:** Yes! Just modify the configuration parameters to match your model:

- `num_paraphrases` - 改写数量 / Number of paraphrases
- `tokens_per_paraphrase` - 每个改写的平均长度 / Average length per paraphrase
- `num_generated_tokens` - 期望生成的长度 / Expected generation length

---

## 🚀 快速开始 / Quick Start

```bash
# 1. 安装依赖 / Install dependencies
pip install jupyter matplotlib numpy

# 2. 启动笔记本 / Start notebook
cd plot/
jupyter notebook flowchart_and_attention_mask_visualization.ipynb

# 3. 运行所有单元格 / Run all cells
# 在 Jupyter 界面中: Cell -> Run All
# In Jupyter interface: Cell -> Run All

# 4. 查看生成的可视化 / View generated visualizations
# 5. 修改 CUSTOM_CONFIG 进行实验 / Modify CUSTOM_CONFIG to experiment
```

---

## 📧 支持 / Support

如果遇到问题，请查看：
If you encounter issues, please check:

1. **依赖安装** - 确保安装了所有必需的包 / **Dependencies** - Ensure all required packages are installed
2. **Python 版本** - 建议使用 Python 3.8+ / **Python version** - Recommended Python 3.8+
3. **测试脚本** - 运行 `test_visualization.py` 验证功能 / **Test script** - Run `test_visualization.py` to verify functions

---

**最后更新 / Last Updated**: 2024-10-28
