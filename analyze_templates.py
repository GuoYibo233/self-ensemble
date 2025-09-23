import pandas as pd

# 读取结果文件
df = pd.read_feather(
    'datasets/myriadlama/qwen2.5_1.5b_it/ensemble_avg-6.feather')

print('=== 分析生成文本中的模板 ===')

# 统计包含不同模板的样本数
adit_count = sum('adit' in gen for gen in df['generation'])
mask_count = sum('[MASK]' in gen for gen in df['generation'])
higher_level_count = sum(
    'higher-level concept' in gen for gen in df['generation'])

print(f'包含"adit"的样本: {adit_count}/{len(df)}')
print(f'包含"[MASK]"的样本: {mask_count}/{len(df)}')
print(f'包含"higher-level concept"的样本: {higher_level_count}/{len(df)}')

print()
print('=== 显示一些不同的生成模式 ===')

# 找一些不同的例子
different_patterns = []
for i, gen in enumerate(df['generation']):
    if 'adit' not in gen:
        different_patterns.append((i, gen))
    if len(different_patterns) >= 5:
        break

for i, (idx, gen) in enumerate(different_patterns):
    print(f'{i+1}. 样本{idx+1}: "{gen}"')

print()
print('=== 检查原始的paraphrases ===')
print('前5个样本的paraphrases:')
for i in range(5):
    if 'paraphrases' in df.columns:
        print(f'样本{i+1}: {df.iloc[i]["paraphrases"]}')

print()
print('=== 检查prompt构造 ===')
print('前3个样本的prompts:')
for i in range(3):
    if 'prompts' in df.columns:
        prompts = df.iloc[i]["prompts"]
        print(f'样本{i+1} prompts数量: {len(prompts)}')
        print(f'  第一个prompt: "{prompts[0][:200]}..."')
        print()
