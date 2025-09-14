# VSCode 逐步调试指南

## 🎯 快速开始

### 1. 启动调试

**方法一：使用快捷键**
```
按 F5 键 → 选择调试配置
```

**方法二：侧边栏调试**
```
点击左侧调试图标 → 选择配置 → 点击绿色播放按钮
```

**方法三：命令面板**
```
Ctrl+Shift+P → 输入 "Debug: Start Debugging"
```

### 2. 可用的调试配置

- `🔍 逐提示模式调试` - 每个改写分别生成
- `🔍 平均集成模式调试` - 平均logits集成  
- `🔍 加权平均集成调试` - 加权平均集成
- `🔍 最大值集成调试` - 最大值集成
- `🔍 加权最大值集成调试` - 加权最大值集成

## 🔍 关键断点位置

### 核心函数断点

1. **主入口函数**
   ```python
   def main():  # 第564行
   ```

2. **数据处理**
   ```python
   def debug_per_prompt_generation():  # 第470行
   def debug_ensemble_generation():    # 第503行
   ```

3. **生成核心**
   ```python
   def debug_single_generation():      # 第138行
   def debug_ensemble_generation_core(): # 第236行
   ```

4. **模型调用**
   ```python
   def __call__(self, input_ids, attention_mask=None):  # MockModel 第63行
   ```

### 推荐断点设置

**初学者断点（了解整体流程）**
```python
# 第564行 - 主函数入口
def main():

# 第470行 - 逐提示生成开始
def debug_per_prompt_generation():

# 第138行 - 单次生成开始  
def debug_single_generation(prompts, max_new_tokens=5):
```

**进阶断点（深入生成细节）**
```python
# 第150行 - 分词后
inputs = tokenizer(prompts, ...)

# 第180行 - 每个生成步骤
for step in range(max_new_tokens):

# 第63行 - 模型前向传播
def __call__(self, input_ids, attention_mask=None):
```

## 🎮 调试操作

### 基本控制

| 快捷键          | 功能     | 说明             |
| --------------- | -------- | ---------------- |
| `F5`            | 继续执行 | 运行到下一个断点 |
| `F10`           | 单步跳过 | 跳过函数调用     |
| `F11`           | 单步进入 | 进入函数内部     |
| `Shift+F11`     | 单步跳出 | 跳出当前函数     |
| `Ctrl+Shift+F5` | 重新启动 | 重新开始调试     |
| `Shift+F5`      | 停止调试 | 结束调试会话     |

### 高级操作

**条件断点**
```python
# 右键断点 → 编辑断点 → 添加条件
# 例如：step == 2 (只在第2步停下)
# 例如：len(prompts) > 1 (只在多个提示时停下)
```

**日志断点**
```python
# 右键断点 → 编辑断点 → 日志消息
# 例如：当前步骤: {step}, logits形状: {logits.shape}
```

## 🔬 调试重点观察

### 1. 数据流转
```python
# 观察这些变量的变化
prompts          # 输入提示
input_ids        # 分词后的ID
logits           # 模型输出
next_token_id    # 选择的下一个token
generated_ids    # 生成的序列
```

### 2. 张量形状
```python
# 关注形状变化
input_ids.shape      # [batch_size, seq_len]
logits.shape         # [batch_size, seq_len, vocab_size]  
attention_mask.shape # [batch_size, seq_len]
```

### 3. 集成过程
```python
# 集成模式特有
all_logits       # 所有改写的logits
ensemble_logits  # 集成后的logits
weights          # 权重（如果是加权方法）
```

## 🛠️ 调试技巧

### 1. 变量监视
在 VARIABLES 面板中监视关键变量：
- `prompts`
- `input_ids` 
- `logits`
- `generated_ids`

### 2. 调试控制台
在 DEBUG CONSOLE 中执行命令：
```python
# 查看张量形状
logits.shape

# 查看前几个概率
torch.softmax(logits[0], dim=-1).topk(3)

# 查看生成的文本
tokenizer.batch_decode(generated_ids)
```

### 3. 监视表达式
添加监视表达式：
```python
logits.shape
torch.softmax(logits[0], dim=-1).max()
len(prompts)
step
```

## 📊 调试示例场景

### 场景1：理解逐提示生成
```python
# 断点设置
1. debug_per_prompt_generation() 开始
2. debug_single_generation() 每次调用
3. 生成循环内部观察 token 选择

# 观察重点
- 每组改写如何分别处理
- 最终如何收集所有预测
```

### 场景2：理解集成策略
```python
# 断点设置  
1. debug_ensemble_generation() 开始
2. debug_ensemble_generation_core() 内部
3. 集成逻辑（平均/最大值）

# 观察重点
- 多组logits如何堆叠
- 不同集成方法的差异
- 最终token如何选择
```

### 场景3：模型调用细节
```python
# 断点设置
1. MockModel.__call__() 方法
2. logits 计算过程
3. 概率分布计算

# 观察重点
- 输入形状变化
- 随机数生成过程
- logits 的统计信息
```

## 🚨 常见问题

### 1. 断点不生效
**解决方案：**
- 确保使用正确的 Python 解释器
- 检查 `justMyCode: false` 设置
- 重启 VSCode

### 2. 变量显示为 `<optimized out>`
**解决方案：**
- 使用 DEBUG CONSOLE 手动查看
- 添加 print 语句
- 使用监视表达式

### 3. 调试速度慢
**解决方案：**
- 减少断点数量
- 使用条件断点
- 关闭不必要的监视

## 💡 最佳实践

1. **从粗到细**：先设置主要流程断点，再深入细节
2. **记录观察**：用注释记录重要发现
3. **对比验证**：不同方法运行结果对比
4. **渐进理解**：从简单示例开始，逐步增加复杂度

---

现在你可以开始逐步调试了！推荐从 `🔍 逐提示模式调试` 开始。