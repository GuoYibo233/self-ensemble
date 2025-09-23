import pandas as pd

# 读取结果文件
df = pd.read_feather(
    'datasets/myriadlama/qwen2.5_1.5b_it/ensemble_avg-6.feather')

print('=== AVG方法生成的文本结果 ===')
print(f'总共 {len(df)} 个样本')
print()

for i in range(20):  # 显示前20个样本
    print(f'{i+1:2d}. 生成文本: "{df.iloc[i]["generation"]}"')
    print(f'    预测答案: "{df.iloc[i]["prediction"]}"')
    print(f'    正确答案: {df.iloc[i]["answers"]}')
    print()
