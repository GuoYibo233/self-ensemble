# FlexAttention测试环境搭建任务

## 任务目标
为FlexAttention项目创建完整的测试和调试框架，确保环境正常工作并提供详细的调试能力。

## 核心需求

### 1. 缓存预热（自动下载资源）
- 下载并缓存数据集：WebQA和MyriadLAMA（少量样本用于测试）
- 下载并缓存模型：Qwen2.5-3B-Instruct（小模型，快速测试）
- 验证所有资源都保存在：`/net/tokyo100-10g/data/str01_01/y-guo/`
- 创建缓存验证脚本

### 2. 环境验证
- 测试PyTorch 2.5.1的FlexAttention功能
- 测试transformers、pandas、numpy等依赖包
- 测试GPU和CUDA可用性
- 运行端到端最小测试

### 3. 调试框架
- 使用极少数据（2-3个问题）进行快速测试
- 在关键位置添加断点建议
- 创建检查函数来查看中间状态
- 添加详细日志记录

## 需要创建的文件

### `test_cache_warmup.py` - 缓存预热脚本
```python
# 下载小模型和少量数据
# 验证缓存位置正确
# 报告下载进度和存储位置
```

### `test_environment.py` - 环境测试脚本
```python
# 测试所有导入
# 测试FlexAttention API
# 测试模型加载
# 生成环境报告
```

### `test_minimal_run.py` - 最小运行测试
```python
# 只处理2个问题，每个5个paraphrase
# 详细日志记录每一步
# 保存中间结果供检查
# 使用小模型快速执行
```

### `debug_helpers.py` - 调试辅助函数
```python
def inspect_tokens(input_ids, tokenizer):
    """检查tokenization结果"""
    
def inspect_segments(text, positions):
    """检查segment位置"""
    
def inspect_attention_mask(mask):
    """可视化attention mask"""
    
def inspect_generation(logits, tokens):
    """检查生成过程"""
```

### `quick_debug.py` - 快速调试脚本
```python
# 集成所有测试
# 提供断点建议
# 交互式调试模式
```

## 断点建议位置

1. **tokenization后**: 检查input_ids, segment_positions
2. **mask创建后**: 检查FlexAttention mask是否正确
3. **每次生成前**: 检查logits和token选择
4. **attention计算时**: 检查attention权重分布
5. **输出生成后**: 对比期望结果

## 检查函数示例
```python
def debug_step(name, **data):
    print(f"\n🔍 调试检查点: {name}")
    for key, val in data.items():
        if torch.is_tensor(val):
            print(f"  {key}: {val.shape} {val.dtype}")
        else:
            print(f"  {key}: {val}")
```

## 执行要求
- 每个测试2分钟内完成
- 使用小于4GB的模型
- 清晰的中文输出和错误信息
- 优雅的错误处理

## 成功标准
✅ 所有测试通过无错误
✅ 缓存正确保存在用户目录
✅ 调试框架提供清晰的执行洞察
✅ 用户可以轻松定位和修复问题
✅ 文档清晰易懂

请创建这些测试文件，包含完整的错误处理、清晰的文档和有用的调试功能。