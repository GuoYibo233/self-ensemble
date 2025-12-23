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

    # Get newline token id
    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]

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
        
        # Stop if newline token is generated in any sequence
        if (next_token == newline_token_id).any():
            break

    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    # Stop at newline to get only the first line
    new_generated_texts = [gen.split('\n')[0].strip() for gen in generated_texts]
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
    
    # Get newline token id
    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]
    
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
        
        # Stop if newline token is generated in any sequence
        if (next_token == newline_token_id).any():
            break
    
    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    # Stop at newline to get only the first line
    new_generated_texts = [gen.split('\n')[0].strip() for gen in generated_texts]
    return new_generated_texts


def sample_paraphrases_per_item(all_paraphrases, num_manual, num_auto, num_samples):
    """
    Two-stage sampling of paraphrases for each item in the batch:
    1. Randomly sample exactly num_manual from manual paraphrases (indices 0-4) 
       and num_auto from auto paraphrases (indices 5-19)
    2. Randomly sample num_samples from the combined pool of num_manual + num_auto
    
    Each UUID (item) in the batch gets different random sampling for maximum diversity.
    
    Args:
        all_paraphrases: List of paraphrase lists, where all_paraphrases[i][j] is the i-th paraphrase version for the j-th item
        num_manual: Exactly how many manual paraphrases to sample (from indices 0-4)
        num_auto: Exactly how many automatic paraphrases to sample (from indices 5-19)
        num_samples: Number of paraphrases to sample from the pool
    
    Returns:
        List of lists: sampled_paraphrases[i][j] is the i-th sampled paraphrase for the j-th item in batch
    """
    batch_size = len(all_paraphrases[0])
    num_paraphrase_versions = len(all_paraphrases)
    
    manual_indices = list(range(5))  # Indices 0-4 are manual
    auto_indices = list(range(5, num_paraphrase_versions))  # Indices 5+ are automatic
    
    # Result structure: [num_samples][batch_size]
    sampled_paraphrases = []
    
    # Process each item in the batch separately for different sampling
    for item_idx in range(batch_size):
        # Stage 1: Randomly sample X manual and Y auto paraphrases
        selected_manual = np.random.choice(manual_indices, size=min(num_manual, len(manual_indices)), replace=False)
        selected_auto = np.random.choice(auto_indices, size=min(num_auto, len(auto_indices)), replace=False)
        
        # Combine into candidate pool
        candidate_pool = list(selected_manual) + list(selected_auto)
        
        # Stage 2: Randomly sample Z from the candidate pool
        final_indices = np.random.choice(candidate_pool, size=min(num_samples, len(candidate_pool)), replace=False)
        
        # Extract the selected paraphrases for this item
        for sample_idx, paraphrase_idx in enumerate(final_indices):
            if sample_idx >= len(sampled_paraphrases):
                sampled_paraphrases.append([])
            sampled_paraphrases[sample_idx].append(all_paraphrases[paraphrase_idx][item_idx])
    
    return sampled_paraphrases


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
    parser.add_argument("--num_ensemble", type=int, default=6, help="Number of models to ensemble. Only used for 'max' and 'avg' methods.")
    parser.add_argument("--num_manual", type=int, default=5, help="Number of manual paraphrases to sample (from indices 0-4). Default: 5")
    parser.add_argument("--num_auto", type=int, default=5, help="Number of automatic paraphrases to sample (from indices 5-19). Default: 5")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of paraphrases to sample from the pool of manual+auto paraphrases. Default: 5")
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
        # Validate parameter constraints
        if args.num_manual < 0 or args.num_auto < 0 or args.num_samples < 0:
            print("ERROR: num_manual, num_auto, and num_samples must be non-negative.")
            exit(1)
        
        if args.num_samples > args.num_manual + args.num_auto:
            print(f"ERROR: num_samples ({args.num_samples}) cannot exceed num_manual + num_auto ({args.num_manual + args.num_auto}).")
            exit(1)
        
        if args.num_manual > 5:
            print("WARNING: num_manual is greater than 5, but only indices 0-4 are manual. Capping at 5.")
            args.num_manual = 5
        
        # Generate output filename based on sampling parameters
        filename = f"ensemble_{args.method}-m{args.num_manual}a{args.num_auto}s{args.num_samples}.feather"
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
        for uuids, answers, all_paraphrases in tqdm(dataloader):
            # Use sampling function to select paraphrases
            # Each UUID gets different random sampling of X manual + Y auto, then Z from pool
            sampled_paraphrases = sample_paraphrases_per_item(
                all_paraphrases, 
                args.num_manual, 
                args.num_auto, 
                args.num_samples
            )
            all_paraphrases = sampled_paraphrases
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
            
            sample_count += 1
            if args.max_samples and sample_count >= args.max_samples:
                print(f"Reached maximum sample limit: {args.max_samples}")
                break
        
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