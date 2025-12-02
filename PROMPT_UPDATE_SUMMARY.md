# Prompt Format Update Summary

## 更新内容

已成功更新 `myriadlama_flex_attention_generate.py` 中的 `construct_prompt_new_format` 函数及其所有调用位置。

## 新的函数行为

### 返回值
函数现在返回一个列表而不是单个字符串：
```python
parts = construct_prompt_new_format(instruction, few_shot_examples, question_paraphrases)
# parts[0] = 共有部分 (shared part)
# parts[1:] = 各个 paraphrase 部分
```

### 共有部分格式 (parts[0])
```
{instruction}

Q: {fs1_para1} A: {fs1_answer}
Q: {fs2_para1} A: {fs2_answer}
Q: {fs3_para1} A: {fs3_answer}

```
- Instruction 后有一个换行
- 每个 few-shot 示例只使用第一个 paraphrase
- 每个 few-shot 在同一行：`Q: ... A: ...`
- Few-shot 部分后有一个换行

### Paraphrase 部分格式 (parts[1:])
每个 paraphrase 是一个独立的字符串：
```
Q: {main_question_paraphrase} A:
```

## 示例输出

```python
parts = [
    # parts[0] - 共有部分
    """You are a helpful assistant. Answer the following question.

Q: What is the capital of France? A: Paris
Q: Who invented relativity? A: Einstein
Q: What is the largest planet? A: Jupiter
""",
    
    # parts[1] - 第 1 个 paraphrase
    "Q: What is X? A:",
    
    # parts[2] - 第 2 个 paraphrase
    "Q: Define X A:",
    
    # parts[3] - 第 3 个 paraphrase
    "Q: X refers to? A:"
]
```

## 组合成完整 prompt

```python
# 方法 1: 简单组合
full_prompt = parts[0] + "\n".join(parts[1:])

# 方法 2: 访问各部分
shared_part = parts[0]
para_parts = parts[1:]
```

## 更新的调用位置

### 1. FlexAttention 生成循环 (约第 1293 行)
```python
# 旧代码
prompt = construct_prompt_new_format(
    instruction,
    few_shot_examples,
    selected_templates
)

# 新代码
prompt_parts = construct_prompt_new_format(
    instruction,
    few_shot_examples,
    selected_templates
)
prompt = prompt_parts[0] + "\n".join(prompt_parts[1:])
```

### 2. Baseline 生成循环 (约第 1428 行)
```python
# 旧代码
prompt = construct_prompt_new_format(
    instruction,
    few_shot_examples,
    [paraphrase]
)

# 新代码
prompt_parts = construct_prompt_new_format(
    instruction,
    few_shot_examples,
    [paraphrase]
)
prompt = prompt_parts[0] + prompt_parts[1]  # shared + single para
```

## 其他修复

1. 修复了不完整的导入语句：
   ```python
   # 旧: from transformers.models.llama
   # 新: from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
   ```

2. 删除了空的函数定义：
   ```python
   # 删除了: def patch_attentiond_model(model, mask_mod):
   ```

## 测试

运行 `test_prompt_format.py` 验证：
```bash
python3 test_prompt_format.py
```

所有测试通过 ✓

## 注意事项

- 函数现在只使用每个 few-shot 示例的**第一个 paraphrase**
- 返回值从字符串变为列表，需要在调用处组合
- 保持了与原有 prompt 格式的兼容性
- 不影响 `parse_prompt_segments_with_metadata_new_format` 等其他函数
