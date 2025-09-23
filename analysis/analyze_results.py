#!/usr/bin/env python3
"""
分析MyriadLAMA数据集上的ensemble生成结果。

使用已有的utils.py中的partial_match函数计算准确率。
"""

from utils import partial_match_scores, partial_match
import os
import pandas as pd
import sys
sys.path.append('../')


def analyze_ensemble_results():
    """分析ensemble结果并计算准确率"""

    # 设置路径 - 针对qwen2.5_1.5b_it模型
    root = "../datasets/myriadlama/qwen2.5_1.5b_it"

    print("=== MyriadLAMA Ensemble Results Analysis ===")
    print(f"模型: qwen2.5_1.5b_it")
    print(f"数据路径: {root}")
    print()

    def get_ensemble_scores(path):
        """计算指定ensemble文件的准确率"""
        if not os.path.exists(path):
            return -1

        try:
            df = pd.read_feather(path)
            print(f"文件 {os.path.basename(path)} 加载成功，包含 {len(df)} 个样本")

            # 检查列名
            print(f"  列名: {list(df.columns)}")

            # 处理answer_lemmas - 确保格式正确
            if 'answer_lemmas' in df.columns:
                df["answer_lemmas"] = df["answer_lemmas"].apply(
                    lambda xs: [list(x) for x in xs])
                answers = df["answer_lemmas"].tolist()
            else:
                print(f"  警告: 文件中没有 'answer_lemmas' 列")
                return -1

            # 计算准确率
            if 'predict_lemma' in df.columns:
                predictions = df['predict_lemma'].tolist()
                avg_acc = partial_match_scores(predictions, answers)

                # 显示一些样例
                print(f"  前3个样例:")
                for i in range(min(3, len(df))):
                    pred = predictions[i]
                    ans = answers[i]
                    match = partial_match(pred, ans, False)
                    print(
                        f"    样例 {i+1}: 预测={pred[:5]}... 答案={ans[0][:5] if ans else []}... 匹配={match}")

                return avg_acc
            else:
                print(f"  警告: 文件中没有 'predict_lemma' 列")
                return -1

        except Exception as e:
            print(f"  错误: 无法处理文件 {path}: {e}")
            return -1

    # 分析不同ensemble方法的结果
    results = {}

    # 测试不同数量的ensemble
    for idx in range(2, 11):
        print(f"\n--- Ensemble数量: {idx} ---")

        # avg方法
        avg_fn = os.path.join(root, f"ensemble_avg-{idx}.feather")
        avg_acc = get_ensemble_scores(avg_fn)

        # max方法
        max_fn = os.path.join(root, f"ensemble_max-{idx}.feather")
        max_acc = get_ensemble_scores(max_fn)

        # weighted方法（如果存在）
        weighted_avg_fn = os.path.join(
            root, f"ensemble_weighted_avg-{idx}.feather")
        weighted_avg_acc = get_ensemble_scores(weighted_avg_fn)

        weighted_max_fn = os.path.join(
            root, f"ensemble_weighted_max-{idx}.feather")
        weighted_max_acc = get_ensemble_scores(weighted_max_fn)

        # 保存结果
        results[idx] = {
            'avg': avg_acc,
            'max': max_acc,
            'weighted_avg': weighted_avg_acc,
            'weighted_max': weighted_max_acc
        }

        # 显示结果
        print(f"结果 - avg: {avg_acc:.3f}, max: {max_acc:.3f}, " +
              f"weighted_avg: {weighted_avg_acc:.3f}, weighted_max: {weighted_max_acc:.3f}")

    # 汇总报告
    print("\n=== 汇总报告 ===")
    print("Ensemble数量 | AVG方法 | MAX方法 | Weighted_AVG | Weighted_MAX")
    print("-" * 65)

    for idx in range(2, 11):
        if idx in results:
            r = results[idx]
            print(f"     {idx:2d}      | {r['avg']:7.3f} | {r['max']:7.3f} | " +
                  f"   {r['weighted_avg']:7.3f}    |    {r['weighted_max']:7.3f}")

    return results


if __name__ == "__main__":
    results = analyze_ensemble_results()
