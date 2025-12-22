from pdb import set_trace
import os
import random
import itertools
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import warnings

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add GYB_self-ensemble directory to path to import its dataset module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GYB_self-ensemble'))

from constants import MODEL_PATHs
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

nlp = None
num_parts = 8

def init_spacy():
    global nlp
    nlp = spacy.load("en_core_web_lg")

def lemmaize_predicts(predict):
    global nlp
    doc = nlp(predict)
    return [token.lemma_.lower() for token in doc]

def lemmaize_chunk(chunk):
    predict_lemmas = []
    answer_lemmas = []
    for prediction, answers in tqdm(zip(chunk["prediction"], chunk["answers"]), total=len(chunk)):
        predict_lemmas.append(lemmaize_predicts(prediction))
        answer_lemmas.append([lemmaize_predicts(ans) for ans in answers])
    return predict_lemmas, answer_lemmas

def append_lemmas(df, results):
    all_predict_lemmas = []
    all_answer_lemmas = []
    for predict_lemmas, answer_lemmas in results:
        all_predict_lemmas.extend(predict_lemmas)
        all_answer_lemmas.extend(answer_lemmas)
    df["predict_lemma"] = pd.Series(all_predict_lemmas, dtype=object)
    df["answer_lemmas"] = pd.Series(all_answer_lemmas, dtype=object)
    return df

def single_generation(prompts, max_new_tokens=10):
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
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    generated = None
    max_new_tokens = 10

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


def sample_paraphrases_per_item(all_paraphrases, num_paraphrases, num_samples, uuids, repeat_paras=False):
    """
    Sample paraphrases using the same logic as generate_myriadlama2.py:
    1. Generate all permutations of paraphrase indices (or repeated patterns if repeat_paras=True)
    2. Randomly sample num_samples permutations from all possible combinations
    
    Each UUID gets deterministic random sampling using uuid as seed.
    Same uuid will always produce the same sampling results, matching generate_myriadlama2.py.
    
    Args:
        all_paraphrases: List of paraphrase lists, where all_paraphrases[i][j] is the i-th paraphrase version for the j-th item
        num_paraphrases: Number of paraphrases to select in each sample
        num_samples: Number of different paraphrase combinations to generate per uuid
        uuids: List of uuids for each item in the batch, used as random seeds
        repeat_paras: If True, repeat the same paraphrase multiple times instead of using permutations
    
    Returns:
        List of samples, where each sample is (uuid, sampled_paraphrases_list) --
    """
    batch_size = len(all_paraphrases[0])
    num_paraphrase_versions = len(all_paraphrases)
    
    all_samples = []
    
    # Process each item in the batch separately with deterministic sampling
    for item_idx in range(batch_size):
        uuid = uuids[item_idx]
        # Get all paraphrases for this item
        item_paraphrases = [all_paraphrases[i][item_idx] for i in range(num_paraphrase_versions)]
        
        # Generate all possible combinations
        all_indices = list(range(len(item_paraphrases)))
        if repeat_paras:
            # Repeat same paraphrase: [[0,0], [1,1], [2,2], ...]
            all_sampled_paras = list([[n] * num_paraphrases for n in all_indices])
        else:
            # Use permutations: all ordered selections of num_paraphrases from available paraphrases
            all_sampled_paras = itertools.permutations(all_indices, num_paraphrases)
        
        # Set random seed based on uuid to ensure deterministic sampling (same as generate_myriadlama2.py)
        random.seed(uuid)
        
        # Sample num_samples different combinations
        all_sampled_paras_list = list(all_sampled_paras)
        sampled_combinations = random.sample(all_sampled_paras_list, k=min(num_samples, len(all_sampled_paras_list)))
        
        # For each combination, extract the actual paraphrases
        for paraids in sampled_combinations:
            sampled_paraphrases = [item_paraphrases[i] for i in paraids]
            all_samples.append((uuid, sampled_paraphrases))
    
    return all_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble generation")
    parser.add_argument("--method", type=str, default="per_prompt", choices=["origin", "per_prompt", "max", "avg", "weighted_avg", "weighted_max"],
                        help="Integration method for ensemble generation")
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Path to the pre-trained model.")
    parser.add_argument("--dataset", type=str, required=True, choices=["webqa", "myriadlama"], help="Dataset to use for generating paraphrases.")    
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda).")
    parser.add_argument("--lemmaize", action="store_true", help="normalize predictions and answers to lemmas.")
    parser.add_argument("--indexs", type=str, default=None, help="Indexs of the dataset to use for generation. If None, use all.")
    parser.add_argument("--num_paraphrases", type=int, default=2, help="Number of paraphrases to use in each sample (default: 2)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of different paraphrase combinations to generate per question (default: 5)")
    parser.add_argument("--repeat_paras", action="store_true", help="Repeat the same paraphrase multiple times instead of using permutations")
    parser.add_argument("--max_samples", type=int, default=250, help="Maximum number of samples to process (default: 250, process all)")
    args = parser.parse_args()    

    if args.dataset == "webqa":
        from dataset import WebQADataset
        dataset = WebQADataset(model_name=args.model)
    elif args.dataset == "myriadlama":
        from dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name=args.model)
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa' or 'myriadlama'.")
    if args.model not in MODEL_PATHs:
        raise ValueError(f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}.")
    
    model_path = MODEL_PATHs.get(args.model, args.model)
    
    if args.method == "origin":
        dump_file = f"./results/{args.model}/origin.feather"
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        if os.path.exists(dump_file):
            print(f"File {dump_file} already exists, skipping generation.")
            exit(0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device, torch_dtype="auto")
        tokenizer.pad_token = tokenizer.eos_token
    
        dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
        df = pd.DataFrame(columns=["uuid", "answers", "question", "prompt", "prediction", "generation"])
        few_shot_context = dataset.get_few_shot_examples()
        
        for uuids, answers, all_paraphrases in tqdm(dataloader):
            # Use only the original questions (paraphrase0)
            original_questions = all_paraphrases[0]
            prompts = dataset.construct_prompts(few_shot_context, original_questions)
            generations = single_generation(prompts)
            predictions = [gen.strip().split('\n')[0] for gen in generations]
            
            items = {
                "uuid": uuids,
                "answers": answers,
                "question": original_questions,
                "prompt": prompts,
                "prediction": predictions,
                "generation": generations,
            }
            df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)
        
        # Lemmaize predictions and answers
        chunks = np.array_split(df, num_parts)
        with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
            results = pool.map(lemmaize_chunk, chunks)
        df = append_lemmas(df, results)
        
        df.to_feather(dump_file)
        csv_file = dump_file.replace('.feather', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"Baseline results saved to {dump_file}")
        print(f"CSV file saved to {csv_file}")
        exit(0)
    
    if args.method == "per_prompt":
        dump_file = f"./results/{args.model}/per_prompt.feather"
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        if os.path.exists(dump_file):
            print(f"File {dump_file} already exists, skipping generation.")
            exit(0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device, torch_dtype="auto")
        tokenizer.pad_token = tokenizer.eos_token

        dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
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
        csv_file = dump_file.replace('.feather', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {dump_file}")
        print(f"CSV file saved to {csv_file}")
        exit(0)
    else:
        # Generate output filename based on sampling parameters
        filename = f"syncedsample_ensemble_{args.method}-{args.num_paraphrases}paras-{args.num_samples}samples"
        if args.repeat_paras:
            filename += "-repeat"
        filename += ".feather"
        dump_file = f"./results/{args.model}/{filename}"
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        
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
            csv_file = dump_file.replace('.feather', '.csv')
            df.to_csv(csv_file, index=False)
            print(f"Lemmatized results saved to {dump_file}")
            print(f"CSV file saved to {csv_file}")
            exit(0)

        if os.path.exists(dump_file):
            print(f"File {dump_file} already exists, skipping generation.")
            exit(0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device, torch_dtype="auto")
        tokenizer.pad_token = tokenizer.eos_token

        dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
        if args.method.startswith("weighted_"):
            conf_df = pd.read_feather(os.path.join(dataset.dataset_root, "confidence.feather"))

        df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation","correctness"])
        print(f"Ensemble generation with method: {args.method}")
        if args.max_samples:
            print(f"Processing maximum {args.max_samples} samples")
        few_shot_context = dataset.get_few_shot_examples()
        sample_count = 0
        
        # First, collect all samples using the same logic as generate_myriadlama2.py
        all_samples = []
        for uuids, answers, all_paraphrases in tqdm(dataloader, desc="Preparing samples"):
            # Use sampling function to select paraphrases
            # Same uuid will always produce the same sampling results (matching generate_myriadlama2.py)
            samples = sample_paraphrases_per_item(
                all_paraphrases, 
                args.num_paraphrases, 
                args.num_samples,
                uuids,
                args.repeat_paras
            )
            # Add answers to each sample
            for uuid, sampled_paraphrases in samples:
                idx = uuids.index(uuid)
                all_samples.append((uuid, answers[idx], sampled_paraphrases))
            
            if args.max_samples and len(all_samples) >= args.max_samples:
                all_samples = all_samples[:args.max_samples]
                break
        
        print(f"Total samples to process: {len(all_samples)}")
        
        # Process each sample
        for uuid, answer, sampled_paraphrases in tqdm(all_samples, desc="Generating"):
            all_prompts = []
            confidences = [] if args.method.startswith("weighted_") else None
            
            # For ensemble generation, treat each paraphrase as a separate prompt
            for para in sampled_paraphrases:
                if args.method.startswith("weighted_"):
                    if confidences is None:
                        confidences = []
                    _sdf = conf_df[conf_df["paraphrase"] == para]
                    if len(_sdf) > 0:
                        confidences.append(float(_sdf["confidence"].values[0]))
                    else:
                        confidences.append(1.0)  # Default confidence
                prompts = dataset.construct_prompts(few_shot_context, [para])
                all_prompts.append(prompts)
            
            generations = ensemble_generation(all_prompts, integration_method=args.method, weights=[confidences] if confidences else None)
            predictions = [gen.strip().split('\n')[0] for gen in generations]
            
            items = {
                "uuid": [uuid],
                "paraphrases": [sampled_paraphrases],
                "prompts": [all_prompts],
                "answers": [answer],
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
        csv_file = dump_file.replace('.feather', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {dump_file}")
        print(f"CSV file saved to {csv_file}")