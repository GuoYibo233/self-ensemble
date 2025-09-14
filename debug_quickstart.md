# 🚀 快速启动 VSCode 调试

## 立即开始

### 1️⃣ 最简单方式
```
按 F5 → 选择 "🔍 逐提示模式调试"
```

### 2️⃣ 推荐断点位置

**🎯 新手入门（4个关键断点）**
```python
# 1. 主函数入口 - 了解程序开始
def main():                    # 第604行左右

# 2. 生成函数开始 - 了解单次生成流程  
def debug_single_generation(): # 第178行左右

# 3. 生成循环开始 - 了解逐步生成过程
for step in range(max_new_tokens):  # 第210行左右

# 4. 模型调用 - 了解模型如何预测
def __call__(self, input_ids, attention_mask=None):  # 第103行左右
```

### 3️⃣ 观察重点变量

在 **VARIABLES** 面板中关注：
```python
prompts          # 输入的提示文本
input_ids        # 分词后的数字序列
logits           # 模型输出的原始分数
next_token       # 选择的下一个token
generated        # 已生成的完整序列
```

### 4️⃣ 使用调试控制台

在 **DEBUG CONSOLE** 中试试这些命令：
```python
# 查看张量形状
logits.shape

# 查看概率分布前3名
import torch
probs = torch.softmax(logits, dim=-1)
probs.topk(3)

# 查看生成的文本
tokenizer.batch_decode(generated)

# 查看当前步骤
step
```

## 🎯 分步调试建议

### 第一轮：整体流程理解
1. 在 `main()` 设置断点
2. 在 `debug_single_generation()` 设置断点  
3. 按 F5 连续执行，观察大的流程

### 第二轮：生成过程深入
1. 在生成循环 `for step in range(max_new_tokens):` 设置断点
2. 按 F10 单步执行，观察每步变化
3. 重点看 `logits` 和 `next_token` 的变化

### 第三轮：模型细节
1. 在 `MockModel.__call__()` 设置断点
2. 观察输入输出的张量变化
3. 理解随机生成 logits 的过程

## 🔧 常用快捷键

| 按键            | 功能     | 使用场景       |
| --------------- | -------- | -------------- |
| `F5`            | 继续运行 | 跳到下一个断点 |
| `F10`           | 单步跳过 | 不进入函数内部 |
| `F11`           | 单步进入 | 进入函数看细节 |
| `Shift+F11`     | 跳出函数 | 回到调用处     |
| `Ctrl+Shift+F5` | 重新开始 | 重新调试       |

## 🚨 调试助手功能

代码中已添加了调试助手，会自动显示：

### 📊 张量信息
```
📊 input_ids 信息:
   形状: torch.Size([2, 13])
   数据类型: torch.int64
   设备: cpu
   数值范围: [0, 999]
   均值: 456.2341
```

### 🔍 调试断点标记
```
🔍 生成函数开始 - 步骤 0
==================================================
```

### 🎯 生成步骤总结
```
🎯 生成步骤 1
------------------------------
选择的token: [622, 215]
已生成序列: [[622], [215]]
```

## 🎮 实际操作步骤

### 开始调试
1. 打开 `debug_generate.py`
2. 在第 604 行 `def main():` 左侧点击设置断点（红点）
3. 按 `F5` 选择 "🔍 逐提示模式调试"
4. 程序会停在断点处

### 逐步执行
1. 按 `F10` 单步执行，观察右侧变量变化
2. 在 VARIABLES 面板展开 `prompts` 看输入数据
3. 继续按 `F10` 直到进入生成循环

### 深入观察
1. 当到达 `debug_single_generation()` 时，按 `F11` 进入
2. 观察 `inputs` 变量的分词结果
3. 在生成循环中，重点观察 `logits` 和 `next_token`

现在就开始你的调试之旅吧！🚀