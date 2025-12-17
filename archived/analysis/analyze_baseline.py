#!/usr/bin/env python3
"""
Analysis script for baseline generation results.

This script analyzes baseline results and compares them with ensemble methods.

Usage:
    python analysis/analyze_baseline.py --dataset webqa --model llama3.2_3b_it
    python analysis/analyze_baseline.py --dataset myriadlama --model qwen2.5_7b_it --compare
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import partial_match, partial_match_scores


def analyze_baseline_origin(dataset_root):
    """
    Analyze Baseline 1: Origin results.
    
    Baseline 1 uses only the original question (attention mode baseline).
    """
    print("\n" + "="*70)
    print("Baseline 1 Analysis: Origin (Attention Mode Baseline)")
    print("="*70)
    
    baseline_file = os.path.join(dataset_root, "baseline_origin.feather")
    
    if not os.path.exists(baseline_file):
        print(f"❌ Baseline origin results not found: {baseline_file}")
        print(f"   Please run: python baseline_generate.py --method origin --dataset <dataset>")
        return None
    
    print(f"✅ Loading baseline origin results: {baseline_file}")
    df = pd.read_feather(baseline_file)
    
    # Calculate accuracy
    if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
        df["answer_lemmas"] = df["answer_lemmas"].apply(
            lambda xs: [list(x) for x in xs] if isinstance(xs, list) else xs
        )
        answers = df["answer_lemmas"].tolist()
        predictions = df['predict_lemma'].tolist()
        
        acc = partial_match_scores(predictions, answers)
        print(f"\n📊 Baseline 1 (Origin) Accuracy: {acc:.3f}")
    else:
        print("\n⚠️  Lemmatized results not available.")
        acc = None
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Unique questions: {df['uuid'].nunique()}")
    
    # Show example generations
    print(f"\n🔍 Sample Generations (first 3):")
    for i in range(min(3, len(df))):
        print(f"\n  Sample {i+1}:")
        print(f"    Question: {df.iloc[i]['question'][:60]}...")
        print(f"    Answer: {df.iloc[i]['answers']}")
        print(f"    Prediction: {df.iloc[i]['prediction']}")
    
    print("="*70)
    return acc


def analyze_baseline_per_prompt(dataset_root):
    """
    Analyze Baseline 2: Per-prompt results.
    
    Baseline 2 generates with each paraphrase separately (second baseline for attention mode).
    """
    print("\n" + "="*70)
    print("Baseline 2 Analysis: Per-Prompt (Attention Mode Second Baseline)")
    print("="*70)
    
    baseline_file = os.path.join(dataset_root, "baseline_per_prompt.feather")
    
    if not os.path.exists(baseline_file):
        print(f"❌ Baseline per_prompt results not found: {baseline_file}")
        print(f"   Please run: python baseline_generate.py --method per_prompt --dataset <dataset>")
        return None
    
    print(f"✅ Loading baseline per_prompt results: {baseline_file}")
    df = pd.read_feather(baseline_file)
    
    # Calculate accuracy
    if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
        df["answer_lemmas"] = df["answer_lemmas"].apply(
            lambda xs: [list(x) for x in xs] if isinstance(xs, list) else xs
        )
        answers = df["answer_lemmas"].tolist()
        predictions = df['predict_lemma'].tolist()
        
        acc = partial_match_scores(predictions, answers)
        print(f"\n📊 Baseline 2 (Per-Prompt) Overall Accuracy: {acc:.3f}")
        
        # Calculate per-paraphrase statistics
        unique_uuids = df['uuid'].unique()
        num_paraphrases = len(df) // len(unique_uuids)
        print(f"\n📈 Per-Paraphrase Statistics:")
        print(f"   Number of paraphrases per question: {num_paraphrases}")
        
        # Group by UUID and calculate accuracy for each paraphrase
        paraphrase_accs = []
        for i in range(num_paraphrases):
            subset = df[df.index % num_paraphrases == i]
            subset_answers = subset["answer_lemmas"].tolist()
            subset_predictions = subset['predict_lemma'].tolist()
            subset_acc = partial_match_scores(subset_predictions, subset_answers)
            paraphrase_accs.append(subset_acc)
            print(f"   Paraphrase {i}: {subset_acc:.3f}")
        
        avg_acc = sum(paraphrase_accs) / len(paraphrase_accs)
        print(f"\n   Average accuracy across paraphrases: {avg_acc:.3f}")
    else:
        print("\n⚠️  Lemmatized results not available.")
        acc = None
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Unique questions: {df['uuid'].nunique()}")
    
    # Show example generations
    print(f"\n🔍 Sample Generations (first question, all paraphrases):")
    first_uuid = df['uuid'].iloc[0]
    first_question_rows = df[df['uuid'] == first_uuid]
    
    for i, row in enumerate(first_question_rows.itertuples()):
        print(f"\n  Paraphrase {i}:")
        print(f"    Paraphrase: {row.paraphrase[:60]}...")
        print(f"    Prediction: {row.prediction}")
    
    print("="*70)
    return acc


def compare_with_ensemble_methods(dataset_root):
    """
    Compare baseline results with ensemble methods.
    """
    print("\n" + "="*70)
    print("Comparison: Baselines vs Ensemble Methods")
    print("="*70)
    
    results = {}
    
    # Load baseline results
    baseline_origin_file = os.path.join(dataset_root, "baseline_origin.feather")
    if os.path.exists(baseline_origin_file):
        df = pd.read_feather(baseline_origin_file)
        if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
            df["answer_lemmas"] = df["answer_lemmas"].apply(
                lambda xs: [list(x) for x in xs] if isinstance(xs, list) else xs
            )
            answers = df["answer_lemmas"].tolist()
            predictions = df['predict_lemma'].tolist()
            acc = partial_match_scores(predictions, answers)
            results["Baseline 1 (Origin)"] = acc
    
    baseline_per_prompt_file = os.path.join(dataset_root, "baseline_per_prompt.feather")
    if os.path.exists(baseline_per_prompt_file):
        df = pd.read_feather(baseline_per_prompt_file)
        if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
            df["answer_lemmas"] = df["answer_lemmas"].apply(
                lambda xs: [list(x) for x in xs] if isinstance(xs, list) else xs
            )
            answers = df["answer_lemmas"].tolist()
            predictions = df['predict_lemma'].tolist()
            acc = partial_match_scores(predictions, answers)
            results["Baseline 2 (Per-Prompt)"] = acc
    
    # Load ensemble method results
    for method in ["avg", "max", "weighted_avg", "weighted_max"]:
        for num_ensemble in [5, 6]:
            ensemble_file = os.path.join(dataset_root, f"ensemble_{method}-{num_ensemble}.feather")
            if os.path.exists(ensemble_file):
                df = pd.read_feather(ensemble_file)
                if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
                    df["answer_lemmas"] = df["answer_lemmas"].apply(
                        lambda xs: [list(x) for x in xs] if isinstance(xs, list) else xs
                    )
                    answers = df["answer_lemmas"].tolist()
                    predictions = df['predict_lemma'].tolist()
                    acc = partial_match_scores(predictions, answers)
                    results[f"Ensemble {method}-{num_ensemble}"] = acc
                break  # Use first available num_ensemble
    
    # Load FlexAttention results
    for num_paraphrases in [5, 6]:
        flex_file = os.path.join(dataset_root, f"flex_attention-{num_paraphrases}.feather")
        if os.path.exists(flex_file):
            df = pd.read_feather(flex_file)
            if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
                df["answer_lemmas"] = df["answer_lemmas"].apply(
                    lambda xs: [list(x) for x in xs] if isinstance(xs, list) else xs
                )
                answers = df["answer_lemmas"].tolist()
                predictions = df['predict_lemma'].tolist()
                acc = partial_match_scores(predictions, answers)
                results[f"FlexAttention-{num_paraphrases}"] = acc
            break  # Use first available num_paraphrases
    
    # Print comparison table
    if results:
        print("\n📊 Accuracy Comparison:")
        print(f"{'Method':<30} {'Accuracy':>10}")
        print("-" * 42)
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for method, acc in sorted_results:
            print(f"{method:<30} {acc:>10.3f}")
        
        # Calculate improvements
        if "Baseline 1 (Origin)" in results:
            baseline_acc = results["Baseline 1 (Origin)"]
            print(f"\n📈 Improvements over Baseline 1 (Origin):")
            for method, acc in sorted_results:
                if method != "Baseline 1 (Origin)":
                    improvement = acc - baseline_acc
                    pct_improvement = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0
                    print(f"   {method:<30} {improvement:+.3f} ({pct_improvement:+.1f}%)")
    else:
        print("\n⚠️  No results available for comparison.")
        print("   Generate baseline and ensemble results first.")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze baseline generation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all baselines
  python analysis/analyze_baseline.py --dataset webqa --model llama3.2_3b_it
  
  # Compare with ensemble methods
  python analysis/analyze_baseline.py --dataset webqa --model llama3.2_3b_it --compare
        """
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["webqa", "myriadlama"],
        help="Dataset name"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g., llama3.2_3b_it, qwen2.5_7b_it)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare baseline results with ensemble methods"
    )
    
    args = parser.parse_args()
    
    # Construct dataset root path
    dataset_root = f"datasets/{args.dataset}/{args.model}"
    
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset root not found: {dataset_root}")
        print(f"   Please check the dataset and model names")
        return
    
    print("="*70)
    print("Baseline Results Analysis")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Dataset root: {dataset_root}")
    
    # Analyze baselines
    origin_acc = analyze_baseline_origin(dataset_root)
    per_prompt_acc = analyze_baseline_per_prompt(dataset_root)
    
    # Compare with ensemble methods if requested
    if args.compare:
        compare_with_ensemble_methods(dataset_root)
    
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)


if __name__ == "__main__":
    main()
