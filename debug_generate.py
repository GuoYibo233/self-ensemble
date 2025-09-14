"""
调试脚本：简化版 generate.py 用于本地运行和理解流程
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import torch
from pdb import set_trace

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 调试辅助函数
# =============================================================================


def debug_tensor_info(tensor, name="tensor"):
    """打印张量的详细信息，便于调试"""
    print(f"\n📊 {name} 信息:")
    print(f"   形状: {tensor.shape}")
    print(f"   数据类型: {tensor.dtype}")
    print(f"   设备: {tensor.device}")
    if tensor.numel() > 0:
        print(
            f"   数值范围: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        # 只对浮点类型计算均值
        if tensor.dtype.is_floating_point:
            print(f"   均值: {tensor.mean().item():.4f}")
        else:
            print(f"   均值: {tensor.float().mean().item():.4f} (转换为浮点)")
        if len(tensor.shape) <= 2:
            print(f"   前5个值: {tensor.flatten()[:5].tolist()}")
    print()
    return tensor


def debug_breakpoint(msg="调试断点", step=None):
    """设置带信息的调试断点"""
    if step is not None:
        print(f"\n🔍 {msg} - 步骤 {step}")
    else:
        print(f"\n🔍 {msg}")
    print("=" * 50)
    # 可以在这里设置断点或添加更多调试信息


def debug_generation_step(step, input_ids, logits, next_tokens, generated_so_far=None):
    """调试生成步骤的详细信息"""
    print(f"\n🎯 生成步骤 {step}")
    print("-" * 30)
    debug_tensor_info(input_ids, "input_ids")
    debug_tensor_info(logits, "logits")
    print(f"选择的token: {next_tokens}")
    if generated_so_far is not None:
        print(f"已生成序列: {generated_so_far}")
    print()

# =============================================================================

# 模拟词形还原


def mock_lemmaize(text):
    """简化版词形还原，只做基本处理"""
    logger.info(f"词形还原输入: '{text}'")
    # 简单处理：小写、分词、去掉标点
    result = [word.lower().strip('.,?!;:()[]{}""\'') for word in text.split()]
    logger.info(f"词形还原结果: {result}")
    return result

# 模拟模型和tokenizer


class MockModel:
    def __init__(self, device="auto"):
        # 自动选择设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 词汇表大小为1000
        self.vocab_size = 1000
        logger.info(f"初始化模拟模型 (vocab_size=1000, device={self.device})")

        if self.device == "cuda":
            logger.info(f"GPU信息: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def __call__(self, input_ids, attention_mask=None):
        # 确保输入在正确的设备上
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        logger.info(f"模型接收输入形状: {input_ids.shape}, 设备: {input_ids.device}")

        # 返回随机logits，形状为 [batch_size, seq_len, vocab_size]
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len,
                             self.vocab_size, device=self.device)

        # 创建一个Result对象，模拟Hugging Face模型输出
        class Result:
            def __init__(self, logits):
                self.logits = logits

        logger.info(f"模型返回logits形状: {logits.shape}")
        return Result(logits)


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 1
        self.pad_token_id = 1
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        logger.info("初始化模拟分词器")

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, padding_side="left", return_attention_mask=True):
        logger.info(f"分词器接收 {len(texts)} 个文本")
        if isinstance(texts, str):
            texts = [texts]

        # 模拟分词：每个token是5-15个随机ID
        input_ids = []
        for text in texts:
            # 文本长度决定token数量，最少5个token
            length = max(5, min(15, len(text) // 10))
            # 随机生成token IDs (范围2-999，保留0和1作为特殊token)
            ids = np.random.randint(2, 1000, size=length).tolist()
            input_ids.append(ids)

        # 找出最长序列并进行padding
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = []
        attention_mask = []

        for ids in input_ids:
            padding_length = max_len - len(ids)
            if padding_side == "left":
                padded = [self.pad_token_id] * padding_length + ids
                mask = [0] * padding_length + [1] * len(ids)
            else:
                padded = ids + [self.pad_token_id] * padding_length
                mask = [1] * len(ids) + [0] * padding_length
            padded_ids.append(padded)
            attention_mask.append(mask)

        # 转换为tensor
        input_ids_tensor = torch.tensor(padded_ids)
        attention_mask_tensor = torch.tensor(attention_mask)

        logger.info(f"分词结果形状: {input_ids_tensor.shape}")

        # 返回类似HF tokenizer的输出
        result = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor if return_attention_mask else None
        }

        return result

    def batch_decode(self, token_ids, skip_special_tokens=True):
        logger.info(f"解码 token_ids 形状: {token_ids.shape}")
        # 模拟解码：将token ID转换为文本
        results = []
        for ids in token_ids:
            # 把token ID转成字符
            text = " ".join([f"[token_{id.item()}]" for id in ids])
            results.append(text)
        logger.info(f"解码样例: '{results[0]}'")
        return results

# 简化版单词生成


def debug_single_generation(prompts, max_new_tokens=5):
    """单词生成的调试版本，最多生成5个token"""
    logger.info(f"=== 开始单词生成 ===")
    logger.info(f"输入: {len(prompts)} 个提示, 最大生成 {max_new_tokens} 个token")

    # 🔍 调试断点 1: 生成开始
    debug_breakpoint("生成函数开始", 0)

    # 设置模型配置
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"设置 pad_token_id = {tokenizer.pad_token_id}")

    # 分词
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="left",
        return_attention_mask=True,
    )
    logger.info(f"分词后输入形状: input_ids={inputs['input_ids'].shape}")

    # 🔍 调试断点 2: 分词完成
    debug_tensor_info(inputs['input_ids'], "分词后的input_ids")

    generated = None

    # 逐步生成
    for step in range(max_new_tokens):
        logger.info(f"\n--- 生成步骤 {step+1}/{max_new_tokens} ---")

        # 🔍 调试断点 3: 每个生成步骤开始
        debug_breakpoint("生成步骤开始", step+1)

        # 【监控点】当前输入状态
        current_seq_len = inputs["input_ids"].shape[1]
        logger.info(f"当前序列长度: {current_seq_len}")
        logger.info(
            f"当前输入形状: input_ids={inputs['input_ids'].shape}, attention_mask={inputs['attention_mask'].shape}")

        # 使用模型预测下一个token
        with torch.no_grad():
            # 获取最后一个位置的logits
            logits = model(
                inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).logits[:, -1, :]
            logger.info(f"获取logits形状: {logits.shape}")

            # 🔍 调试断点 4: 获得模型输出
            debug_tensor_info(logits, f"步骤{step+1}的logits")

            # 【监控点】查看logits的统计信息
            logger.info(
                f"logits统计: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")

            # 【监控点】查看概率分布
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=3, dim=-1)
            for batch_idx in range(min(2, probs.shape[0])):  # 只显示前2个样本
                logger.info(
                    f"样本 {batch_idx} 前3个概率: {top_probs[batch_idx].tolist()}")
                logger.info(
                    f"样本 {batch_idx} 对应token: {top_indices[batch_idx].tolist()}")

            # 选择概率最高的token
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
            logger.info(f"选择的下一个token: {next_token.squeeze().tolist()}")

            # 🔍 调试断点 5: token选择完成
            debug_tensor_info(next_token, f"步骤{step+1}选择的token")

        # 将新token添加到输入中
        old_input_shape = inputs["input_ids"].shape
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], next_token], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(next_token)], dim=1
        )
        new_input_shape = inputs["input_ids"].shape
        logger.info(f"输入形状变化: {old_input_shape} -> {new_input_shape}")

        # 🔍 调试信息：生成步骤总结
        debug_generation_step(
            step+1,
            inputs["input_ids"],
            logits,
            next_token.squeeze().tolist(),
            generated.tolist() if generated is not None else None
        )

        # 【监控点】显示当前生成的完整序列
        current_generated_ids = inputs["input_ids"][:,
                                                    old_input_shape[1]:]  # 只显示新生成的部分
        logger.info(f"到目前为止生成的token IDs: {current_generated_ids.tolist()}")

        # 保存生成的token
        if generated is None:
            generated = next_token
            logger.info(
                f"初始化生成序列: 形状={generated.shape}, 内容={generated.tolist()}")
        else:
            old_generated_shape = generated.shape
            generated = torch.cat([generated, next_token], dim=1)
            logger.info(f"扩展生成序列: {old_generated_shape} -> {generated.shape}")
            logger.info(f"完整生成序列: {generated.tolist()}")

    # 解码生成的token
    generated_texts = tokenizer.batch_decode(
        generated, skip_special_tokens=True)
    new_generated_texts = [gen.strip() for gen in generated_texts]

    logger.info(f"=== 生成完成 ===")
    for i, (prompt, gen) in enumerate(zip(prompts, new_generated_texts)):
        logger.info(f"示例 {i+1}:")
        logger.info(f"  提示: {prompt[:50]}..." if len(
            prompt) > 50 else f"  提示: {prompt}")
        logger.info(f"  生成: {gen}")

    return new_generated_texts

# 简化版集成生成


def debug_ensemble_generation(prompt_sets, integration_method="max", weights=None):
    """集成生成的调试版本"""
    logger.info(f"\n=== 开始集成生成 (方法: {integration_method}) ===")
    logger.info(f"输入: {len(prompt_sets)} 组提示, 每组 {len(prompt_sets[0])} 个样本")

    if weights is not None:
        logger.info(f"使用权重: 形状={np.array(weights).shape}")

    tokenizer.pad_token_id = tokenizer.eos_token_id

    generated = None
    max_new_tokens = 5  # 简化为最多生成5个token

    # 为每组提示进行分词
    all_inputs = []
    for i, prompts in enumerate(prompt_sets):
        logger.info(f"处理第 {i+1} 组提示")
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
            return_attention_mask=True,
        )
        all_inputs.append(inputs)

    all_input_ids = [inputs["input_ids"] for inputs in all_inputs]
    all_attention_masks = [inputs["attention_mask"] for inputs in all_inputs]

    # 逐步生成
    for step in range(max_new_tokens):
        logger.info(f"\n--- 集成生成步骤 {step+1}/{max_new_tokens} ---")
        logits_set = []

        # 获取每组提示的logits
        with torch.no_grad():
            for idx, prompts in enumerate(prompt_sets):
                logger.info(f"计算第 {idx+1} 组提示的logits")
                input_ids = all_input_ids[idx]
                attention_mask = all_attention_masks[idx]
                logits = model(input_ids, attention_mask=attention_mask).logits[
                    :, -1, :
                ]
                logits_set.append(logits)

            # 堆叠所有logits
            logits_set = torch.stack(logits_set)
            logger.info(f"堆叠所有logits: {logits_set.shape}")

        # 根据集成方法选择下一个token
        if integration_method == "avg":
            logger.info("使用平均值集成")
            avg_logits = logits_set.mean(dim=0)
            logger.info(f"平均logits形状: {avg_logits.shape}")

            # 【监控点】显示平均前后的logits对比
            logger.info(f"原始logits统计 - 形状: {logits_set.shape}")
            for i in range(min(2, logits_set.shape[0])):
                logger.info(
                    f"  组 {i}: min={logits_set[i].min().item():.4f}, max={logits_set[i].max().item():.4f}, mean={logits_set[i].mean().item():.4f}")
            logger.info(
                f"平均后logits统计: min={avg_logits.min().item():.4f}, max={avg_logits.max().item():.4f}, mean={avg_logits.mean().item():.4f}")

            next_token = torch.argmax(avg_logits, dim=-1).unsqueeze(1)

        elif integration_method == "max":
            logger.info("使用最大值集成")
            # 转换为概率
            probs_set = logits_set.softmax(dim=-1)
            logger.info(f"概率集合形状: {probs_set.shape}")

            # 【监控点】显示概率统计
            for i in range(min(2, probs_set.shape[0])):
                logger.info(
                    f"组 {i} 概率统计: min={probs_set[i].min().item():.6f}, max={probs_set[i].max().item():.6f}")

            # 取每个位置的最大概率
            max_probs = probs_set.max(dim=0).values
            logger.info(f"最大概率形状: {max_probs.shape}")
            logger.info(
                f"最大概率统计: min={max_probs.min().item():.6f}, max={max_probs.max().item():.6f}")
            next_token = torch.argmax(max_probs, dim=-1).unsqueeze(1)

        elif integration_method == "weighted_avg":
            if weights is None:
                raise ValueError(
                    "Weights must be provided for weighted_avg integration.")

            logger.info("使用加权平均集成")
            # 转换权重为tensor
            weights_tensor = torch.tensor(weights).float()
            logger.info(f"原始权重形状: {weights_tensor.shape}")
            logger.info(f"原始权重值: {weights_tensor.tolist()}")

            # 归一化权重
            weight_sums = weights_tensor.sum(dim=0)
            weights_tensor = weights_tensor / weight_sums.unsqueeze(0)
            logger.info(f"归一化后权重: {weights_tensor.tolist()}")
            logger.info(
                f"权重和验证 (应该都是1.0): {weights_tensor.sum(dim=0).tolist()}")

            # 【监控点】显示加权过程
            logger.info(
                f"logits_set形状: {logits_set.shape}, weights形状: {weights_tensor.shape}")
            expanded_weights = weights_tensor.unsqueeze(-1)
            logger.info(f"扩展权重形状: {expanded_weights.shape}")

            # 加权平均
            weighted_logits = (logits_set * expanded_weights).sum(dim=0)
            logger.info(f"加权logits形状: {weighted_logits.shape}")
            logger.info(
                f"加权logits统计: min={weighted_logits.min().item():.4f}, max={weighted_logits.max().item():.4f}")
            next_token = torch.argmax(weighted_logits, dim=-1).unsqueeze(1)

        else:
            raise ValueError(f"不支持的集成方法: {integration_method}")

        logger.info(f"选择的下一个token: {next_token.squeeze().tolist()}")

        # 更新所有输入
        for i in range(len(all_input_ids)):
            all_input_ids[i] = torch.cat([all_input_ids[i], next_token], dim=1)
            all_attention_masks[i] = torch.cat(
                [all_attention_masks[i], torch.ones_like(next_token)], dim=1
            )

        # 保存生成的token
        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)

        logger.info(f"当前生成序列形状: {generated.shape}")

    # 解码生成的token
    generated_texts = tokenizer.batch_decode(
        generated, skip_special_tokens=True)
    new_generated_texts = [gen.strip() for gen in generated_texts]

    logger.info(f"=== 集成生成完成 ===")
    for i, gen in enumerate(new_generated_texts):
        logger.info(f"样本 {i+1} 生成结果: {gen}")

    return new_generated_texts

# 创建模拟数据集


class MockDataset:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dataset_root = "./mock_data"
        os.makedirs(self.dataset_root, exist_ok=True)
        logger.info(
            f"创建模拟数据集: model_name={model_name}, root={self.dataset_root}")

    def get_dataloader(self, batch_size=2, shuffle=False):
        """返回一个简单的数据加载器，只有两个样本"""
        # 创建三个样本的数据
        uuids = [["id1"], ["id2"]]
        answers = [[["Paris"]], [["Jupiter"]]]

        # 为每个样本创建多组改写
        all_paraphrases = [
            [
                ["What is the capital of France?",
                    "Tell me the capital city of France."],
                ["What is Paris known for?", "Why is Paris famous?"]
            ],
            [
                ["What is the largest planet?",
                    "Which planet is biggest in our solar system?"],
                ["Tell me about Jupiter.", "What makes Jupiter special?"]
            ]
        ]

        logger.info(
            f"创建模拟数据加载器: {len(uuids)} 个样本, 每个有 {len(all_paraphrases[0])} 组改写")
        # 返回一个只迭代一次的迭代器
        return [(uuids, answers, all_paraphrases)]

    def get_few_shot_examples(self):
        """返回少量示例的上下文"""
        examples = [
            "Q: What is the capital of Spain?\nA: Madrid",
            "Q: What is the largest ocean?\nA: Pacific Ocean"
        ]
        logger.info(f"获取少样本示例: {len(examples)} 个")
        return examples

    def construct_prompts(self, few_shot_context, paraphrases):
        """根据少样本上下文和改写构建提示"""
        prompts = []
        for paraphrase in paraphrases:
            # 将上下文和改写组合成完整提示
            prompt = "\n\n".join(few_shot_context)
            prompt += f"\n\nQ: {paraphrase}\nA: "
            prompts.append(prompt)
        logger.info(f"构建提示: {len(prompts)} 个")
        return prompts


# 设置全局模型和分词器
logger.info("初始化全局模型和分词器")
# 注意：这里只是为了向后兼容，实际初始化在main函数中
model = None
tokenizer = None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="调试生成过程")
    parser.add_argument(
        "--method",
        type=str,
        default="per_prompt",
        choices=["per_prompt", "max", "avg", "weighted_avg"],
        help="集成方法",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2,
        help="示例数量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="运行设备：auto自动选择，cpu强制CPU，cuda强制GPU",
    )
    args = parser.parse_args()

    logger.info(
        f"启动调试脚本: method={args.method}, device={args.device}")

    # 设置全局模型和分词器
    logger.info("初始化全局模型和分词器")
    model = MockModel(device=args.device)
    tokenizer = MockTokenizer()

    # 创建模拟数据集
    dataset = MockDataset(model_name="mock_model")

    if args.method == "per_prompt":
        logger.info("\n\n===== 逐提示生成模式 =====")
        # 获取模拟数据
        for uuids, answers, all_paraphrases in dataset.get_dataloader():
            logger.info(f"处理批次: {len(uuids)} 个样本")

            # 保存中间结果
            predictions_by_paraphrase = []

            few_shot_context = dataset.get_few_shot_examples()

            # 对每个改写组分别生成
            for paraphrase_group in all_paraphrases:
                logger.info(f"\n处理改写组: {len(paraphrase_group)} 个改写")

                for paraphrases in paraphrase_group:
                    # 构建提示
                    prompts = dataset.construct_prompts(
                        few_shot_context, paraphrases)
                    logger.info(f"构建了 {len(prompts)} 个提示")

                    # 生成
                    generations = debug_single_generation(prompts)

                    # 提取预测结果
                    predictions = [
                        gen.split()[0] if gen else "" for gen in generations]
                    predictions_by_paraphrase.append(predictions)

            # 显示所有结果
            logger.info("\n===== 生成结果摘要 =====")
            for i, (uuid, answer) in enumerate(zip(uuids, answers)):
                logger.info(f"样本 {i+1} (ID: {uuid[0]}):")
                logger.info(f"  正确答案: {answer[0]}")

                for j, preds in enumerate(predictions_by_paraphrase):
                    if i < len(preds):
                        logger.info(f"  改写组 {j+1} 预测: {preds[i]}")

    else:
        logger.info(f"\n\n===== 集成生成模式: {args.method} =====")

        # 获取模拟数据
        for uuids, answers, all_paraphrases in dataset.get_dataloader():
            logger.info(f"处理批次: {len(uuids)} 个样本")

            few_shot_context = dataset.get_few_shot_examples()

            # 准备不同改写组的提示
            all_prompts = []
            for paraphrase_group in all_paraphrases[0]:  # 只使用第一个样本的改写
                prompts = dataset.construct_prompts(
                    few_shot_context, paraphrase_group)
                all_prompts.append(prompts)

            # 如果是加权方法，创建模拟权重
            if args.method.startswith("weighted_"):
                logger.info("创建模拟权重")
                weights = []
                for i in range(len(all_prompts)):
                    # 创建随机权重
                    group_weights = np.random.uniform(
                        0.5, 1.0, size=len(all_prompts[0]))
                    weights.append(group_weights.tolist())
                logger.info(f"权重: {weights}")
            else:
                weights = None

            # 进行集成生成
            generations = debug_ensemble_generation(
                all_prompts,
                integration_method=args.method,
                weights=weights
            )

            # 提取预测结果
            predictions = [
                gen.split()[0] if gen else "" for gen in generations]

            # 显示结果
            logger.info("\n===== 集成生成结果 =====")
            for i, (uuid, answer, pred) in enumerate(zip(uuids, answers, predictions)):
                if i >= len(predictions):
                    break
                logger.info(f"样本 {i+1} (ID: {uuid[0]}):")
                logger.info(f"  正确答案: {answer[0]}")
                logger.info(f"  集成预测: {pred}")

            # 执行词形还原示例
            logger.info("\n===== 词形还原示例 =====")
            for i, pred in enumerate(predictions):
                if i >= len(predictions):
                    break
                lemmas = mock_lemmaize(pred)
                logger.info(f"样本 {i+1} 词形还原:")
                logger.info(f"  原始: {pred}")
                logger.info(f"  结果: {lemmas}")

    logger.info("\n调试脚本执行完成!")
