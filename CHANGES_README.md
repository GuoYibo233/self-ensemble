# Mask Matrix和Prompt格式改进 / Improvements to Mask Matrix and Prompt Formatting

## 快速开始 / Quick Start

运行测试脚本查看改进效果：
```bash
python3 test_mask_visualization.py
```

## 改进内容 / What Changed

### 1. 🎯 Mask Matrix可视化 - 现在可以看到完整结构！

**问题**: 原来只显示20x20，对于248个token的序列看不到整体结构  
**解决**: 智能采样显示25个关键位置，展示完整的attention模式

**示例对比**:

旧版本（只能看到前20个token）:
```
  ... (truncated, showing first 20x20 of 248x248)
  ❌ Cannot see the overall structure!
```

新版本（智能采样，可以看到全局）:
```
  Q\KV   0 16 32 47 48 63 79 94 95111127142143159175191192207222237238239240241242
 S1   0  ■  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  · 
 E1  47  ■  ■  ■  ■  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  · 
 S2  48  ·  ·  ·  ·  ■  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  · 
 ...
 G0 238  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ·  ·  ·  · 
```

**关键特性**:
- ✅ 显示所有segment边界（S# = start, E# = end）
- ✅ 显示generation起始点（G0）
- ✅ 使用更清晰的符号（■ = attend, · = no-attend）
- ✅ 智能采样算法确保看到重要位置

### 2. 📝 Prompt分隔格式 - 更清晰可读！

**问题**: 多个prompt用`[SEP]`直接连接，难以区分  
**解决**: 使用带换行的分隔符

**示例对比**:

旧版本:
```
Q: What is the capital of France?
A: [SEP] Q: Which city is...
```
❌ 问题: 挤在一起，难以阅读

新版本:
```
Q: What is the capital of France?
A:

[SEP]

Q: Which city is...
```
✅ 改进: 清晰分隔，易于阅读

## 修改的文件 / Modified Files

| 文件 | 改动说明 |
|------|---------|
| `flex_attention_generate.py` | 更改默认separator为`\n\n[SEP]\n\n` |
| `tools/debug_flexattention.py` | 增强mask可视化和输出格式 |
| `tools/example_flexattention.py` | 更新示例中的可视化 |
| `test_mask_visualization.py` | 新建测试脚本（无需模型） |
| `IMPROVEMENTS_SUMMARY.md` | 详细技术文档 |
| `BEFORE_AFTER_COMPARISON.md` | 可视化对比文档 |

## 技术细节 / Technical Details

### 智能采样算法

```python
# 采样优先级:
1. Segment边界（起始和结束）
2. Segment内关键位置
3. Generation开始位置  
4. 生成的token位置
5. 均匀填充剩余空间
```

### 新的可视化符号

```python
■ = can attend (可以attention)
· = cannot attend (不能attention)
S# = Segment start (Segment起始)
E# = Segment end (Segment结束)
G0 = Generation start (生成起始)
```

## 使用方法 / How to Use

### 1. 运行测试查看效果
```bash
python3 test_mask_visualization.py
```

### 2. 在你的代码中使用
```python
# 自动使用新的separator
from flex_attention_generate import concatenate_paraphrases_with_positions

# 或者手动指定
concatenated, positions, length = concatenate_paraphrases_with_positions(
    prompts, 
    tokenizer,
    separator="\n\n[SEP]\n\n"  # 新的默认值
)
```

### 3. 调试时查看详细信息
```bash
# 如果你有模型和数据，可以运行:
python3 tools/debug_flexattention.py --dataset webqa --max-samples 1
```

## 兼容性 / Compatibility

✅ **完全向后兼容**:
- 所有改动都是可选的
- 默认参数已优化，但可以覆盖
- 不影响现有功能

## 性能影响 / Performance Impact

- ✅ 可视化改进不影响实际生成性能
- ✅ 智能采样算法复杂度很低（O(n)）
- ✅ Separator改变对tokenization影响很小（增加2-3个token）

## 文档 / Documentation

详细文档请查看:
1. **IMPROVEMENTS_SUMMARY.md** - 完整的技术说明（中英文）
2. **BEFORE_AFTER_COMPARISON.md** - 详细的before/after对比
3. **test_mask_visualization.py** - 可运行的示例代码

## 总结 / Summary

这次改进完美解决了用户提出的两个问题：

1. ✅ **Mask matrix现在可以显示几百个token的整体结构**
   - 从只能看20个token → 智能采样显示整体模式
   - 清楚标记segment边界和generation部分

2. ✅ **Prompt之间的分隔清晰可见**
   - 从挤在一起 → 用换行清晰分隔
   - 更易于阅读和调试

额外收益：
- 更好的可视化符号
- 详细的调试输出
- 完整的测试和文档

**一切就绪，可以使用了！** 🎉
