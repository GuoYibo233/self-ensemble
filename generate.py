"""
集成式生成（self-ensemble）。

支持多种融合策略：
- per_prompt：逐提示独立生成（不融合），便于对比。
- avg：对多个提示的下一步 logits 取平均后再 argmax。
- max：对多个提示的下一步概率取逐类最大后再 argmax。
- weighted_avg：按权重（通常来源于 confidence）对 logits 加权平均。
- weighted_max：对权重沿样本位置取 argmax，选择对应提示的 logits 再 argmax。

可选对生成/答案进行词形还原（lemmaize）以便后续匹配与统计。
"""

from pdb import set_trace
import os
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from constants import MODEL_PATHs
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

nlp = None
num_parts = 8

def init_spacy():
    """多进程初始化：加载 spaCy 大模型用于词形还原。"""
    global nlp
    nlp = spacy.load("en_core_web_lg")

def lemmaize_predicts(predict):
    """将文本转为词形还原后的小写 token 列表。"""
    global nlp
    doc = nlp(predict)
    return [token.lemma_.lower() for token in doc]

def lemmaize_chunk(chunk):
    """对 DataFrame 分块批量做词形还原，返回 (pred_lemmas, answer_lemmas)。"""
    predict_lemmas = []
    answer_lemmas = []
    for prediction, answers in tqdm(zip(chunk["prediction"], chunk["answers"]), total=len(chunk)):
        predict_lemmas.append(lemmaize_predicts(prediction))
        answer_lemmas.append([lemmaize_predicts(ans) for ans in answers])
    return predict_lemmas, answer_lemmas

def append_lemmas(df, results):
    """将多进程返回的词形还原结果写回到 DataFrame。"""
    all_predict_lemmas = []
    all_answer_lemmas = []
    for predict_lemmas, answer_lemmas in results:
        all_predict_lemmas.extend(predict_lemmas)
        all_answer_lemmas.extend(answer_lemmas)
    df["predict_lemma"] = pd.Series(all_predict_lemmas, dtype=object)
    df["answer_lemmas"] = pd.Series(all_answer_lemmas, dtype=object)
    return df

def single_generation(prompts, max_new_tokens=20):
    """逐步贪心生成实现（显式步进），便于可控地融合或对比。"""
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompts, return_tensors="pt", 
        padding=True, truncation=True, 
        padding_side='left', return_attention_mask=True).to(model.device)

    generated = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)

        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(next_token)], dim=1)

        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)

    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    new_generated_texts = [gen.strip() for gen in generated_texts]
    return new_generated_texts

@torch.no_grad()
def ensemble_generation(prompt_sets, integration_method="max", weights=None):
    """基于多提示的下一步分布进行 token 级融合生成。

    Args:
        prompt_sets (List[List[str]]): N 组提示，每组长度为 batch。
        integration_method (str): 融合策略，见模块说明。
        weights (List[List[float]]|Tensor|None): 每组提示在每个样本位置的权重（加权策略使用）。
    Returns:
        List[str]: 每个样本一条最终生成文本。
    """
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    generated = None
    max_new_tokens = 20

    all_inputs = []
    for prompts in prompt_sets:
        inputs = tokenizer(
            prompts, return_tensors="pt", 
            padding=True, truncation=True, 
            padding_side='left', return_attention_mask=True).to(model.device)
        all_inputs.append(inputs)

    all_input_ids = [inputs["input_ids"] for inputs in all_inputs]
    all_attention_masks = [inputs["attention_mask"] for inputs in all_inputs]
    for _ in range(max_new_tokens):
        logits_set = []
        with torch.no_grad():
            for idx, prompts in enumerate(prompt_sets):
                input_ids = all_input_ids[idx]
                attention_mask = all_attention_masks[idx]
                logits = model(input_ids, attention_mask=attention_mask).logits[:, -1, :]
                logits_set.append(logits)
            logits_set = torch.stack(logits_set)

        if integration_method == "avg":
            avg_logits = logits_set.mean(dim=0)
            next_token = torch.argmax(avg_logits, dim=-1).unsqueeze(1)
        elif integration_method == "max":    
            max_probs = logits_set.softmax(dim=-1).max(dim=0).values
            next_token = torch.argmax(max_probs, dim=-1).unsqueeze(1)
        elif integration_method == "weighted_avg":
            if weights is None:
                raise ValueError("Weights must be provided for weighted_avg integration.")
            weights = torch.tensor(weights).clone().detach().requires_grad_(False).to(logits_set.device)
            weights = weights / weights.sum(dim=0).unsqueeze(0)
            weighted_logits = (logits_set * weights.unsqueeze(-1)).sum(dim=0)
            next_token = torch.argmax(weighted_logits, dim=-1).unsqueeze(1)
        elif integration_method == "weighted_max":
            if weights is None:
                raise ValueError("Weights must be provided for weighted_max integration.")
            weights = torch.tensor(weights).clone().detach().requires_grad_(False).to(logits_set.device)
            argmax = weights.argmax(dim=0)
            max_logits = logits_set[argmax, torch.arange(logits_set.shape[1])]
            next_token = max_logits.argmax(dim=-1).unsqueeze(1)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
        
        # Take the element-wise min across the two distributions
        # Append next token to input_ids for next round
        for i in range(len(all_input_ids)):            
            all_input_ids[i] = torch.cat([all_input_ids[i], next_token], dim=1)
            all_attention_masks[i] = torch.cat([all_attention_masks[i], torch.ones_like(next_token)], dim=1)

        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)
        generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        new_generated_texts = [gen.strip() for gen in generated_texts]
    return new_generated_texts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble generation")
    parser.add_argument("--method", type=str, default="per_prompt", choices=["per_prompt", "max", "avg", "weighted_avg", "weighted_max"],
                        help="Integration method for ensemble generation")
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Path to the pre-trained model.")
    parser.add_argument("--dataset", type=str, required=True, choices=["webqa", "myriadlama"], help="Dataset to use for generating paraphrases.")    
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model on.")
    parser.add_argument("--lemmaize", action="store_true", help="normalize predictions and answers to lemmas.")
    parser.add_argument("--indexs", type=str, default=None, help="Indexs of the dataset to use for generation. If None, use all.")
    parser.add_argument("--num_ensemble", type=int, default=6, help="Number of models to ensemble. Only used for 'max' and 'avg' methods.")
    args = parser.parse_args()    

    if args.dataset == "webqa":
        from dataset import WebQADataset
        dataset = WebQADataset(model_name=args.model)
    elif args.dataset == "myriadlama":
        from dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name=args.model)
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa' or 'myriadlama'.")

        
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    if args.model not in MODEL_PATHs:
        raise ValueError(f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}.")
    
    model_path = MODEL_PATHs.get(args.model, args.model)
    if args.method == "per_prompt":
        dump_file = f"{dataset.dataset_root}/per_prompt.feather"
        if os.path.exists(dump_file):
            print(f"File {dump_file} already exists, skipping generation.")
            exit(0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device, torch_dtype="auto")
        tokenizer.pad_token = tokenizer.eos_token

        df = pd.DataFrame(columns=["uuid", "answers", "prompts", "predictions"])
        for uuids, answers, all_paraphrases in tqdm(dataloader):
            preds_in_batch = []
            prompts_in_batch = []
            paraphrases_in_batch = []
            generations_in_batch = []
            predictions_in_batch = []

            few_shot_context = dataset.get_few_shot_examples()
            for paraphrases in all_paraphrases:
                paraphrases_in_batch.extend(paraphrases)
                prompts = dataset.construct_prompts(few_shot_context, paraphrases)
                generations = single_generation(prompts)
                predictions = [gen.strip().split('\n')[0] for gen in generations]
                prompts_in_batch.extend(prompts)
                preds_in_batch.extend(predictions)
                generations_in_batch.extend(generations)
                predictions_in_batch.extend(predictions)
            items = {
                "uuid": [uuid[0] for uuid in uuids] * len(all_paraphrases),
                "answers": [ans[0] for ans in answers] * len(all_paraphrases),
                "paraphrase": paraphrases_in_batch,
                "prompt": prompts_in_batch,
                "prediction": predictions_in_batch,
                "generation": generations_in_batch,
            }
            df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)
        df.to_feather(dump_file)
        exit(0)
    else:
        if args.indexs is not None:
            _root = os.path.join(dataset.dataset_root, "diversity")
            os.makedirs(_root, exist_ok=True)
            dump_file = f"{_root}/ensemble_{args.method}-{args.indexs}.feather"
        else:
            dump_file = f"{dataset.dataset_root}/ensemble_{args.method}-{args.num_ensemble}.feather"
        
        print(f"Dump file: {dump_file}")

        if args.lemmaize:
            assert os.path.exists(dump_file), f"Confidence scores file {dump_file} does not exist. Please run the script with --rewrite to create it."
            df = pd.read_feather(dump_file)
            if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
                print(f"Confidence scores already exist in {dump_file}. Use --rewrite to overwrite.")
                exit(0)
            
            chunks = np.array_split(df, num_parts)
            with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
                results = pool.map(lemmaize_chunk, chunks)

            df = append_lemmas(df, results)
            df.to_feather(dump_file)
            exit(0)

        if os.path.exists(dump_file):
            print(f"File {dump_file} already exists, skipping generation.")
            exit(0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device, torch_dtype="auto")
        tokenizer.pad_token = tokenizer.eos_token

        if args.method.startswith("weighted_"):
            conf_df = pd.read_feather(os.path.join(dataset.dataset_root, "confidence.feather"))

        df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation"])
        print(f"Ensemble generation with method: {args.method}")
        few_shot_context = dataset.get_few_shot_examples()
        for uuids, answers, all_paraphrases in tqdm(dataloader):
            if args.indexs is not None:
                all_paraphrases = [all_paraphrases[int(idx)] for idx in args.indexs.split(",")]
            else:
                assert len(all_paraphrases) >= args.num_ensemble, f"Expected at least {args.num_ensemble} paraphrases, found {len(all_paraphrases)}"
                all_paraphrases = all_paraphrases[:args.num_ensemble]
                assert len(all_paraphrases) == args.num_ensemble, f"Expected {args.num_ensemble} paraphrases, found {len(all_paraphrases)}"
            all_paraphrases = [list(paras) for paras in all_paraphrases]
            all_prompts = []
            confidences = [] if args.method.startswith("weighted_") else None
            for paraphrases in all_paraphrases:
                if args.method.startswith("weighted_"):
                    confidences.append([])
                    for para in paraphrases:
                        _sdf = conf_df[conf_df["paraphrase"] == para]
                        # assert len(_sdf) == 1, f"Expected one confidence score for paraphrase '{para}', found {_sdf.shape[0]}"
                        confidences[-1].append(float(_sdf["confidence"].values[0]))
                prompts = dataset.construct_prompts(few_shot_context, paraphrases)
                all_prompts.append(prompts)
            generations = ensemble_generation(all_prompts, integration_method=args.method, weights=confidences)
            predictions = [gen.strip().split('\n')[0] for gen in generations]
            items = {
                "uuid": uuids,
                "paraphrases": list(zip(*all_paraphrases)),
                "prompts": list(zip(*all_prompts)),
                "answers": answers,
                "prediction": predictions,
                "generation": generations,
            }
            df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)
        
        # Split the DataFrame into chunks for multiprocessing
        chunks = np.array_split(df, num_parts)
        with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
            results = pool.map(lemmaize_chunk, chunks)
        df = append_lemmas(df, results)
        
        # Save the DataFrame to a Feather file
        df.to_feather(dump_file)
