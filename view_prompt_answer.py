import pandas as pd

# 读取结果文件
df = pd.read_feather(
    'datasets/myriadlama/qwen2.5_1.5b_it/ensemble_avg-6.feather')

print('=== 完整的 Prompt 和答案对应关系 ===')
print(f'总共 {len(df)} 个样本')
print()

for i in range(5):  # 显示前5个样本的详细信息
    print(f'{"="*60}')
    print(f'样本 {i+1}')
    print(f'{"="*60}')

    # 显示UUID和正确答案
    print(f'UUID: {df.iloc[i]["uuid"]}')
    print(f'正确答案: {df.iloc[i]["answers"]}')
    print()

    # 显示所有6个paraphrases
    paraphrases = df.iloc[i]["paraphrases"]
    print(f'使用的6个paraphrases:')
    for j, para in enumerate(paraphrases):
        print(f'  {j+1}. {para}')
    print()

    # 显示对应的6个prompts
    prompts = df.iloc[i]["prompts"]
    print(f'对应的6个完整prompts:')
    for j, prompt in enumerate(prompts):
        print(f'--- Prompt {j+1} ---')
        print(prompt)
        print()

    # 显示ensemble生成的结果
    print(f'Ensemble生成结果:')
    print(f'  生成文本: "{df.iloc[i]["generation"]}"')
    print(f'  提取答案: "{df.iloc[i]["prediction"]}"')
    print(f'  预测词形还原: {df.iloc[i]["predict_lemma"]}')
    print(f'  答案词形还原: {df.iloc[i]["answer_lemmas"]}')

    print()
    print()
