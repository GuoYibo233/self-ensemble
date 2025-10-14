# Mask Matrix and Prompt Formatting Improvements

## 问题总结 / Problem Summary

根据用户需求，解决了两个主要问题：

1. **Mask Matrix显示问题**: 原来只显示20x20的矩阵，当序列有几百个token时，无法看清attention的整体结构
2. **Prompt输出格式问题**: 多个prompt之间用`[SEP]`直接连接，没有空格或换行，难以区分

## 改进方案 / Improvements

### 1. Mask Matrix可视化增强

**改进前 (Before)**:
- 只显示前20x20的矩阵
- 对于248个token的序列，只能看到开头一小部分
- 无法了解整体的attention结构

**改进后 (After)**:
- 使用智能采样策略，显示约25个关键位置
- 优先包含：
  - 所有segment的起始和结束位置
  - 每个segment内的关键位置
  - 生成部分的token位置
  - 均匀分布的中间位置
- 使用清晰的标记：
  - `S#`: Segment起始位置
  - `E#`: Segment结束位置  
  - `G0`: Generation开始位置
  - `■`: 可以attend
  - `·`: 不能attend

**示例输出**:
```
Mask Matrix (248x248):
  ✅ Showing 25 strategic positions (including segment boundaries)
  Q\KV   0 16 32 47 48 63 79 94 95111127142143159175191192207222237238239240241242
 S1   0  ■  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  · 
 E1  47  ■  ■  ■  ■  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  · 
 S2  48  ·  ·  ·  ·  ■  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  · 
 ...
 G0 238  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ·  ·  ·  · 
```

现在可以清楚地看到：
- 5个segment完全隔离（对角线块状结构）
- 生成的token可以attend到所有之前的token（最后几行全是■）

### 2. Prompt格式改进

**改进前 (Before)**:
```
Q: What is the capital of France?
A: [SEP] Q: Which city is the capital of France?
A: [SEP] Q: France's capital city is called?
A:
```
❌ 问题: 难以区分不同的prompt

**改进后 (After)**:
```
Q: What is the capital of France?
A:

[SEP]

Q: Which city is the capital of France?
A:

[SEP]

Q: France's capital city is called?
A:
```
✅ 改进: 使用`\n\n[SEP]\n\n`作为分隔符，清晰可见

### 3. 调试输出增强

在`debug_flexattention.py`中，新增了更详细的segment显示：

```
Full Sequence with Segment Markers:
======================================================================

[Prompt 1] (positions 0-47):
Q: What is the capital of France?
A:
----------------------------------------------------------------------

[Prompt 2] (positions 48-94):
Q: Which city is the capital of France?
A:
----------------------------------------------------------------------

[Generated Output] (positions 238-247):
Paris
----------------------------------------------------------------------
```

## 修改的文件 / Modified Files

1. **tools/example_flexattention.py**
   - 更新`visualize_mask()`函数，支持智能采样显示

2. **tools/debug_flexattention.py**
   - 更新`visualize_attention_mask()`，使用新的采样策略
   - 更新`debug_concatenation()`，增加segment预览
   - 增强最终输出显示，分segment展示

3. **flex_attention_generate.py**
   - 修改默认separator从`" [SEP] "`改为`"\n\n[SEP]\n\n"`

4. **test_mask_visualization.py** (新建)
   - 完整的测试脚本，对比展示改进效果
   - 无需模型即可运行

## 如何测试 / How to Test

运行测试脚本查看改进效果：

```bash
python3 test_mask_visualization.py
```

这个脚本会展示：
- 旧版本vs新版本的mask matrix可视化对比
- 旧版本vs新版本的separator格式对比
- 完整的改进总结

## 技术细节 / Technical Details

### 智能采样算法

1. 首先添加所有segment边界（起始和结束位置）
2. 在每个segment内添加1-2个采样点
3. 添加generation开始位置
4. 添加最后几个生成的token位置
5. 用二分法填充剩余空间，直到达到max_display限制

### 符号选择

- `■` (BLACK SQUARE): 表示可以attend，视觉上更明显
- `·` (MIDDLE DOT): 表示不能attend，不会太突兀
- 比原来的`✓`和`✗`更紧凑，可以显示更多内容

## 兼容性 / Compatibility

所有改动都是向后兼容的：
- 默认参数值已更新，但可以通过参数覆盖
- 原有的功能逻辑保持不变
- 只是改进了输出格式和可视化效果

## 总结 / Summary

✅ **解决了原始问题**:
1. 现在可以看到几百个token的整体attention结构
2. Prompt之间的分隔清晰可见

✅ **额外改进**:
1. 更好的视觉符号
2. Segment标记（S#/E#/G0）
3. 详细的调试输出
4. 完整的测试脚本

这些改进让FlexAttention的调试和可视化变得更加清晰和有用！
