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
    # 使用 spaCy 的 NLP 管道对输入文本进行处理
    # 输入: predict 是一个字符串，通常是模型生成的预测文本（例如 "Running quickly"）
    # 输出: 返回一个列表，包含输入文本中每个单词的词形还原形式（小写）

    global nlp  # 声明全局变量 nlp，用于 spaCy 的 NLP 管道
    doc = nlp(predict)  # 将输入文本转换为 spaCy 的 Doc 对象，进行分词和词形还原等处理

    # 遍历 Doc 对象中的每个 Token，提取词形还原形式（lemma），并转换为小写
    return [token.lemma_.lower() for token in doc]

def lemmaize_chunk(chunk):
    # This function processes a chunk of data to normalize predictions and answers to their lemmatized forms.
    # Input: chunk - A dictionary containing "prediction" (list of predicted strings) and "answers" (list of lists of answer strings).
    # Output: Two lists - predict_lemmas (lemmatized predictions) and answer_lemmas (lemmatized answers).
    #
    # Example:
    # chunk = {
    #     "prediction": ["Running fast", "Eating apples"],
    #     "answers": [["Ran quickly", "Jogging"], ["Ate an apple", "Chewing"]]
    # }
    # predict_lemmas, answer_lemmas = lemmaize_chunk(chunk)
    # print(predict_lemmas)  # Output: [["run", "fast"], ["eat", "apple"]]
    # print(answer_lemmas)   # Output: [[["run", "quickly"], ["jog"]], [["eat", "an", "apple"], ["chew"]]]

    predict_lemmas = []  # Initialize an empty list to store lemmatized predictions.
    answer_lemmas = []  # Initialize an empty list to store lemmatized answers.

    # Iterate over each prediction and its corresponding answers in the chunk.
    # tqdm is used to display a progress bar for the iteration.
    for prediction, answers in tqdm(zip(chunk["prediction"], chunk["answers"]), total=len(chunk)):
        # Lemmatize the prediction string and append the result to predict_lemmas.
        predict_lemmas.append(lemmaize_predicts(prediction))

        # Lemmatize each answer string in the list of answers and append the results to answer_lemmas.
        answer_lemmas.append([lemmaize_predicts(ans) for ans in answers])

    # Return the lemmatized predictions and answers as two separate lists.
    return predict_lemmas, answer_lemmas

def append_lemmas(df, results):
    # This function appends lemmatized predictions and answers to a DataFrame as new columns.
    # Input: df - A Pandas DataFrame containing the original data.
    #        results - A list of tuples, where each tuple contains lemmatized predictions and answers.
    # Output: The updated DataFrame with new columns "predict_lemma" and "answer_lemmas".
    #
    # Example:
    # df = pd.DataFrame({
    #     "uuid": [1, 2],
    #     "prediction": ["Running fast", "Eating apples"],
    #     "answers": [["Ran quickly", "Jogging"], ["Ate an apple", "Chewing"]]
    # })
    # results = [
    #     (["run", "fast"], [["run", "quickly"], ["jog"]]),
    #     (["eat", "apple"], [["eat", "an", "apple"], ["chew"]])
    # ]
    # df = append_lemmas(df, results)
    # print(df)
    # Output:
    #    uuid       prediction                     answers       predict_lemma                  answer_lemmas
    # 0     1    Running fast  [Ran quickly, Jogging]    [run, fast]  [[run, quickly], [jog]]
    # 1     2  Eating apples  [Ate an apple, Chewing]  [eat, apple]  [[eat, an, apple], [chew]]

    all_predict_lemmas = []
    all_answer_lemmas = []

    # Iterate over the results and extend the lists with lemmatized predictions and answers.
    for predict_lemmas, answer_lemmas in results:
        all_predict_lemmas.extend(predict_lemmas)
        all_answer_lemmas.extend(answer_lemmas)

    # Add the lemmatized predictions and answers as new columns in the DataFrame.
    df["predict_lemma"] = pd.Series(all_predict_lemmas, dtype=object)
    df["answer_lemmas"] = pd.Series(all_answer_lemmas, dtype=object)

    # Return the updated DataFrame.
    return df

def single_generation(prompts, max_new_tokens=20):
    # This function generates text predictions for a list of prompts using a language model.
    # Input: prompts - A list of strings, each representing a prompt for the model.
    #        max_new_tokens - The maximum number of tokens to generate for each prompt.
    # Output: A list of generated text strings, one for each prompt.

    # Set the padding token ID to the end-of-sequence token ID for the tokenizer.
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Disable temperature and top-p sampling for deterministic generation.
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    # Ensure the padding token ID is set in the model's generation configuration.
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    # Tokenize the input prompts into tensors suitable for the model.
    inputs = tokenizer(
        prompts,  # The list of input prompts to tokenize.
        return_tensors="pt",  # Return PyTorch tensors.
        padding=True,  # Pad the sequences to the same length.
        truncation=True,  # Truncate sequences that are too long.
        padding_side='left',  # Add padding tokens to the left side of the sequence.
        return_attention_mask=True  # Include an attention mask to indicate padding positions.
    ).to(model.device)  # Move the tokenized inputs to the same device as the model.

    generated = None  # Initialize the variable to store the generated tokens.

    # Generate tokens iteratively up to the maximum number of new tokens.
    for _ in range(max_new_tokens):
        with torch.no_grad():  # Disable gradient computation for faster inference.
            # Option: Create custom attention mask to manually control attention scores
            # custom_attention_mask = create_custom_attention_mask(inputs["attention_mask"])
            
            # Compute the logits (raw predictions) for the next token.
            logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[:, -1, :]
            # Alternative: Use custom attention mask
            # logits = model(inputs["input_ids"], attention_mask=custom_attention_mask).logits[:, -1, :]
            
            # Select the token with the highest probability as the next token.
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)

        # Append the next token to the input IDs for the next iteration.
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        # Update the attention mask to include the new token.
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(next_token)], dim=1)

        # Store the generated tokens.
        if generated is None:
            generated = next_token  # Initialize the generated tokens with the first token.
        else:
            generated = torch.cat([generated, next_token], dim=1)  # Append the new token to the generated tokens.

    # Decode the generated tokens into text, skipping special tokens.
    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    # Strip leading and trailing whitespace from each generated text.
    new_generated_texts = [gen.strip() for gen in generated_texts]

    # Return the list of generated text strings.
    return new_generated_texts

@torch.no_grad()
def ensemble_generation(prompt_sets, integration_method="max", weights=None):
    # This function performs ensemble text generation by combining predictions from multiple prompt sets.
    # Input: prompt_sets - A list of prompt lists, where each inner list contains prompts for one ensemble member.
    #        integration_method - Method to combine predictions ("max", "avg", "weighted_avg", "weighted_max").
    #        weights - Optional weights for weighted integration methods.
    # Output: A list of generated text strings combining all ensemble predictions.

    # Set the padding token ID to the end-of-sequence token ID for the tokenizer.
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Disable temperature and top-p sampling for deterministic generation.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    # Ensure the padding token ID is set in the model's generation configuration.
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    generated = None  # Initialize the variable to store the generated tokens.
    max_new_tokens = 20  # Maximum number of tokens to generate.

    # Tokenize all prompt sets and store the results.
    all_inputs = []
    for prompts in prompt_sets:  # Iterate through each set of prompts for ensemble members.
        # Tokenize the current set of prompts.
        inputs = tokenizer(
            prompts, return_tensors="pt",  # Return PyTorch tensors.
            padding=True, truncation=True,  # Pad sequences and truncate if necessary.
            padding_side='left', return_attention_mask=True).to(model.device)  # Left padding with attention mask.
        all_inputs.append(inputs)  # Store the tokenized inputs.

    # Extract input IDs and attention masks from all tokenized inputs.
    all_input_ids = [inputs["input_ids"] for inputs in all_inputs]
    all_attention_masks = [inputs["attention_mask"] for inputs in all_inputs]
    
    # Generate tokens iteratively up to the maximum number of new tokens.
    for _ in range(max_new_tokens):
        logits_set = []  # Initialize list to store logits from all ensemble members.
        with torch.no_grad():  # Disable gradient computation for faster inference.
            # Compute logits for each ensemble member.
            # Example: If we have 3 ensemble members with different paraphrases of the same question
            for idx, prompts in enumerate(prompt_sets):
                # idx=0: prompts = ["What is the capital of France?", "What is the capital of Spain?"]
                # idx=1: prompts = ["Which city is France's capital?", "Which city is Spain's capital?"]  
                # idx=2: prompts = ["France's capital city is?", "Spain's capital city is?"]
                
                input_ids = all_input_ids[idx]  # Get input IDs for the current ensemble member.
                # Example: input_ids shape = [2, 15] (2 prompts, 15 tokens each after padding)
                # input_ids[0] = [1, 1841, 338, 278, 7483, 310, 3444, 29973, 0, 0, 0, 0, 0, 0, 0]
                
                attention_mask = all_attention_masks[idx]  # Get attention mask for the current ensemble member.
                # Example: attention_mask shape = [2, 15] (same shape as input_ids)
                # attention_mask[0] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0] (1=real token, 0=padding)
                
                # Compute logits for the next token using the current ensemble member's inputs.
                logits = model(input_ids, attention_mask=attention_mask).logits[:, -1, :]
                # Example: model output shape = [2, 15, 32000] (2 prompts, 15 positions, 32000 vocab size)
                # logits[:, -1, :] extracts the last position logits: shape = [2, 32000]
                # logits[0] = [0.1, -0.5, 2.3, ..., 0.8] (32000 values representing probability scores for each vocab token)
                # The highest value might be at position 3Paris (token for "Paris")
                
                logits_set.append(logits)  # Store the logits for this ensemble member.
                # After idx=0: logits_set = [tensor([2, 32000])]
                # After idx=1: logits_set = [tensor([2, 32000]), tensor([2, 32000])]
                # After idx=2: logits_set = [tensor([2, 32000]), tensor([2, 32000]), tensor([2, 32000])]
                
            logits_set = torch.stack(logits_set)  # Stack logits from all ensemble members into a tensor.
            # Example: Final logits_set shape = [3, 2, 32000] 
            # - 3 ensemble members
            # - 2 prompts per member  
            # - 32000 possible next tokens
            # logits_set[0, 0, :] = logits from ensemble member 0, prompt 0 (What is the capital of France?)
            # logits_set[1, 0, :] = logits from ensemble member 1, prompt 0 (Which city is France's capital?)
            # logits_set[2, 0, :] = logits from ensemble member 2, prompt 0 (France's capital city is?)

        # Apply the specified integration method to combine logits from all ensemble members.
        if integration_method == "avg":
            # Average method: Take the mean of logits across all ensemble members.
            avg_logits = logits_set.mean(dim=0)
            # Select the token with the highest probability from the averaged logits.
            next_token = torch.argmax(avg_logits, dim=-1).unsqueeze(1)
        elif integration_method == "max":    
            # Max method: Convert logits to probabilities and take the maximum probability for each token.
            max_probs = logits_set.softmax(dim=-1).max(dim=0).values
            # Select the token with the highest maximum probability.
            next_token = torch.argmax(max_probs, dim=-1).unsqueeze(1)
        elif integration_method == "weighted_avg":
            # Weighted average method: Combine logits using provided weights.
            if weights is None:
                raise ValueError("Weights must be provided for weighted_avg integration.")
            # Convert weights to tensor and normalize them.
            weights = torch.tensor(weights).clone().detach().requires_grad_(False).to(logits_set.device)
            weights = weights / weights.sum(dim=0).unsqueeze(0)  # Normalize weights to sum to 1.
            # Compute weighted average of logits.
            weighted_logits = (logits_set * weights.unsqueeze(-1)).sum(dim=0)
            # Select the token with the highest probability from the weighted logits.
            next_token = torch.argmax(weighted_logits, dim=-1).unsqueeze(1)
        elif integration_method == "weighted_max":
            # Weighted max method: Select the ensemble member with the highest weight.
            if weights is None:
                raise ValueError("Weights must be provided for weighted_max integration.")
            # Convert weights to tensor.
            weights = torch.tensor(weights).clone().detach().requires_grad_(False).to(logits_set.device)
            # Find the ensemble member with the highest weight for each position.
            argmax = weights.argmax(dim=0)
            # Extract logits from the highest-weighted ensemble member.
            max_logits = logits_set[argmax, torch.arange(logits_set.shape[1])]
            # Select the token with the highest probability from the selected ensemble member.
            next_token = max_logits.argmax(dim=-1).unsqueeze(1)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
        
        # Update input sequences for all ensemble members with the newly generated token.
        # Append next token to input_ids for next round
        for i in range(len(all_input_ids)):            
            # Append the generated token to the input IDs of each ensemble member.
            all_input_ids[i] = torch.cat([all_input_ids[i], next_token], dim=1)
            # Update the attention mask to include the new token (mark as non-padding).
            all_attention_masks[i] = torch.cat([all_attention_masks[i], torch.ones_like(next_token)], dim=1)

        # Store the generated tokens for final output.
        if generated is None:
            generated = next_token  # Initialize the generated tokens with the first token.
        else:
            generated = torch.cat([generated, next_token], dim=1)  # Append the new token to the generated tokens.
        
        # Decode the generated tokens into text for progress tracking (optional).
        generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        new_generated_texts = [gen.strip() for gen in generated_texts]
    
    # Return the final list of generated text strings.
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