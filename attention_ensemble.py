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
    """
    Same as original generate.py
    """
    global nlp
    doc = nlp(predict)
    return [token.lemma_.lower() for token in doc]

def lemmaize_chunk(chunk):
    """
    Same as original generate.py
    """
    predict_lemmas = []
    answer_lemmas = []

    for prediction, answers in tqdm(zip(chunk["prediction"], chunk["answers"]), total=len(chunk)):
        predict_lemmas.append(lemmaize_predicts(prediction))
        answer_lemmas.append([lemmaize_predicts(ans) for ans in answers])

    return predict_lemmas, answer_lemmas

def append_lemmas(df, results):
    """
    Same as original generate.py
    """
    all_predict_lemmas = []
    all_answer_lemmas = []

    for predict_lemmas, answer_lemmas in results:
        all_predict_lemmas.extend(predict_lemmas)
        all_answer_lemmas.extend(answer_lemmas)

    df["predict_lemma"] = pd.Series(all_predict_lemmas, dtype=object)
    df["answer_lemmas"] = pd.Series(all_answer_lemmas, dtype=object)

    return df

def concatenate_paraphrases_with_separator(prompts, tokenizer, separator_token=" [SEP] "):
    """
    将多个paraphrase连接在一起，并记录每个paraphrase的token位置
    
    Args:
        prompts: List of prompt strings (paraphrases)
        tokenizer: Tokenizer to use for tokenization
        separator_token: Token to separate different paraphrases
    
    Returns:
        concatenated_text: 连接后的文本
        segment_positions: List of (start, end) positions for each paraphrase
        total_length: Total token length after concatenation
    """
    # 连接所有paraphrases，用分隔符分开
    concatenated_text = separator_token.join(prompts)
    
    # 计算每个paraphrase的token位置
    segment_positions = []
    current_pos = 0
    
    for i, prompt in enumerate(prompts):
        # 对当前prompt进行tokenization以计算长度
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_tokens)
        
        if i > 0:
            # 加上分隔符的长度
            sep_tokens = tokenizer.encode(separator_token, add_special_tokens=False)
            current_pos += len(sep_tokens)
        
        start_pos = current_pos
        end_pos = current_pos + prompt_length
        segment_positions.append((start_pos, end_pos))
        current_pos = end_pos
    
    # 对完整文本进行tokenization
    full_tokens = tokenizer.encode(concatenated_text, add_special_tokens=True)
    total_length = len(full_tokens)
    
    return concatenated_text, segment_positions, total_length

def create_isolation_attention_mask(segment_positions, total_length, device):
    """
    创建隔离注意力掩码，使不同的paraphrase不能互相关注
    
    Args:
        segment_positions: List of (start, end) positions for each segment
        total_length: Total sequence length
        device: Device to create tensor on
    
    Returns:
        4D attention mask: [1, 1, total_length, total_length]
    """
    # 创建基础的因果掩码（下三角）
    causal_mask = torch.tril(torch.ones(total_length, total_length, device=device))
    
    # 创建段落隔离掩码
    isolation_mask = torch.zeros(total_length, total_length, device=device)
    
    # 为每个段落设置只能内部关注的掩码
    for start, end in segment_positions:
        # 每个段落内部可以互相关注（因果约束下）
        for i in range(start, end):
            for j in range(start, min(i + 1, end)):  # 因果约束：只能关注前面的token
                isolation_mask[i, j] = 1
    
    # 合并因果掩码和隔离掩码
    final_mask = causal_mask * isolation_mask
    
    # 扩展为4D: [batch_size=1, num_heads=1, seq_len, seq_len]
    mask_4d = final_mask.unsqueeze(0).unsqueeze(0)
    
    return mask_4d

@torch.no_grad()
def attention_ensemble_generation(prompts, max_new_tokens=20):
    """
    使用传统4D attention mask进行集成生成
    
    Args:
        prompts: List of paraphrase prompts
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        Generated text string
    """
    # 设置模型参数
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    # 步骤1: 将多个paraphrase连接在一起
    concatenated_text, segment_positions, total_length = concatenate_paraphrases_with_separator(
        prompts, tokenizer, separator_token=" [SEP] "
    )
    
    print(f"Concatenated text: {concatenated_text}")
    print(f"Segment positions: {segment_positions}")
    print(f"Total length: {total_length}")
    
    # 步骤2: 对连接后的文本进行tokenization
    inputs = tokenizer(
        concatenated_text,
        return_tensors="pt",
        truncation=True,
        return_attention_mask=True
    ).to(model.device)
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    generated = None
    
    # 步骤3: 生成循环
    for step in range(max_new_tokens):
        current_length = inputs["input_ids"].shape[1]
        
        # 更新segment_positions以包含新生成的token
        # 新生成的token被视为一个新的段落，可以关注所有前面的内容
        current_segment_positions = segment_positions + [(total_length, current_length)]
        
        # 使用传统4D attention mask进行段落隔离
        custom_mask = create_isolation_attention_mask(
            current_segment_positions, current_length, model.device
        )
        
        # 扩展mask到正确的head数量
        num_heads = getattr(model.config, 'num_attention_heads', 32)
        batch_size = inputs["input_ids"].shape[0]
        mask_4d = custom_mask.repeat(batch_size, num_heads, 1, 1)
        
        # 应用原始的padding mask
        original_mask = inputs["attention_mask"]
        for i in range(batch_size):
            for j in range(current_length):
                if original_mask[i, j] == 0:  # 如果是padding
                    mask_4d[i, :, :, j] = 0
                    mask_4d[i, :, j, :] = 0
        
        # 使用自定义attention mask进行前向传播
        with torch.no_grad():
            logits = model(inputs["input_ids"], attention_mask=mask_4d).logits[:, -1, :]
        
        # 选择下一个token
        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
        
        # 更新输入序列
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(next_token)], dim=1)
       
        set_trace()
        
        # 存储生成的token
        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)
    
    # 解码生成的文本
    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return generated_texts[0].strip()

def single_generation_fallback(prompts, max_new_tokens=20):
    """
    Fallback function for when FlexAttention is not available
    """
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side='left',
        return_attention_mask=True
    ).to(model.device)

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
    return [gen.strip() for gen in generated_texts]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Attention-based ensemble generation")
    parser.add_argument("--method", type=str, default="attention_ensemble", 
                        help="Generation method")
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", 
                        help="Path to the pre-trained model.")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["webqa", "myriadlama"], 
                        help="Dataset to use for generating paraphrases.")    
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to run the model on.")
    parser.add_argument("--lemmaize", action="store_true", 
                        help="normalize predictions and answers to lemmas.")
    parser.add_argument("--indexs", type=str, default=None, 
                        help="Indexs of the dataset to use for generation. If None, use all.")
    parser.add_argument("--num_paraphrases", type=int, default=5, 
                        help="Number of paraphrases to use for attention ensemble.")
    args = parser.parse_args()    

    # Load dataset
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
    
    # Set up output file
    if args.indexs is not None:
        _root = os.path.join(dataset.dataset_root, "diversity")
        os.makedirs(_root, exist_ok=True)
        dump_file = f"{_root}/attention_ensemble-{args.indexs}.feather"
    else:
        dump_file = f"{dataset.dataset_root}/attention_ensemble-{args.num_paraphrases}.feather"
    
    print(f"Dump file: {dump_file}")

    # Handle lemmatization mode
    if args.lemmaize:
        assert os.path.exists(dump_file), f"File {dump_file} does not exist. Please run without --lemmaize first."
        df = pd.read_feather(dump_file)
        if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
            print(f"Lemmatized data already exists in {dump_file}.")
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

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device, torch_dtype="auto")
    tokenizer.pad_token = tokenizer.eos_token

    # Check model configuration
    print(f"🔍 Model information:")
    print(f"   PyTorch version: {torch.__version__}")
    if hasattr(model.config, 'num_attention_heads'):
        print(f"   Model attention heads: {model.config.num_attention_heads}")
    print(f"   Using traditional 4D attention mask for paraphrase isolation")

    # Main generation loop
    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation"])
    print(f"Attention ensemble generation with {args.num_paraphrases} paraphrases")
    few_shot_context = dataset.get_few_shot_examples()
    
    for uuids, answers, all_paraphrases in tqdm(dataloader):
        if args.indexs is not None:
            selected_paraphrases = [all_paraphrases[int(idx)] for idx in args.indexs.split(",")]
        else:
            # Select the specified number of paraphrases
            selected_paraphrases = all_paraphrases[:args.num_paraphrases]
        
        batch_predictions = []
        batch_generations = []
        
        # Process each question in the batch
        for i, paraphrases in enumerate(zip(*selected_paraphrases)):
            # Construct prompts for all paraphrases of this question
            prompts = []
            for paraphrase in paraphrases:
                prompt = dataset.construct_prompts(few_shot_context, [paraphrase])
                prompts.append(prompt[0])  # Get the single prompt
            
            # Use attention ensemble generation
            try:
                generation = attention_ensemble_generation(prompts, max_new_tokens=20)
                prediction = generation.strip().split('\n')[0]
            except Exception as e:
                print(f"Attention ensemble failed: {e}, using fallback")
                # Fallback to first paraphrase
                fallback_generations = single_generation_fallback([prompts[0]], max_new_tokens=20)
                generation = fallback_generations[0]
                prediction = generation.strip().split('\n')[0]
            
            batch_predictions.append(prediction)
            batch_generations.append(generation)
        
        # Store results
        items = {
            "uuid": uuids,
            "paraphrases": list(zip(*selected_paraphrases)),
            "answers": answers,
            "prediction": batch_predictions,
            "generation": batch_generations,
        }
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)
    
    # Save results
    df.to_feather(dump_file)
    print(f"✅ Results saved to {dump_file}")