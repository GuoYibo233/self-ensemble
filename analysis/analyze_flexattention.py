#!/usr/bin/env python3
"""
Analysis script for FlexAttention generation results.

This script analyzes the output from flex_attention_generate.py and compares
it with traditional ensemble methods.

Usage:
    python analysis/analyze_flexattention.py --dataset myriadlama --model qwen2.5_7b_it
    python analysis/analyze_flexattention.py --dataset webqa --model llama3.2_3b_it
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import partial_match, partial_match_scores


def analyze_flexattention_results(dataset_root, num_paraphrases=5):
    """
    Analyze FlexAttention generation results.
    
    Args:
        dataset_root: Path to dataset results directory
        num_paraphrases: Number of paraphrases used (default: 5)
    """
    
    print("="*70)
    print("FlexAttention Generation Analysis")
    print("="*70)
    print(f"Dataset root: {dataset_root}")
    print(f"Number of paraphrases: {num_paraphrases}")
    print()
    
    # Load FlexAttention results
    flex_file = os.path.join(dataset_root, f"flex_attention-{num_paraphrases}.feather")
    
    if not os.path.exists(flex_file):
        print(f"❌ FlexAttention results not found: {flex_file}")
        print(f"   Please run: python flex_attention_generate.py --dataset <dataset> --num_paraphrases {num_paraphrases}")
        return
    
    print(f"✅ Loading FlexAttention results: {flex_file}")
    df_flex = pd.read_feather(flex_file)
    
    # Process lemmas if available
    if "predict_lemma" in df_flex.columns and "answer_lemmas" in df_flex.columns:
        df_flex["answer_lemmas"] = df_flex["answer_lemmas"].apply(
            lambda xs: [list(x) for x in xs] if isinstance(xs, list) else xs
        )
        answers = df_flex["answer_lemmas"].tolist()
        predictions = df_flex['predict_lemma'].tolist()
        
        flex_acc = partial_match_scores(predictions, answers)
        print(f"\n📊 FlexAttention Accuracy (with lemmatization): {flex_acc:.3f}")
    else:
        print("\n⚠️  Lemmatized results not available. Run with --lemmaize flag to generate them.")
        print("   Example: python flex_attention_generate.py --dataset <dataset> --lemmaize")
        flex_acc = None
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(df_flex)}")
    print(f"   Unique UUIDs: {df_flex['uuid'].nunique()}")
    
    # Show some example generations
    print(f"\n🔍 Sample Generations (first 3):")
    for i in range(min(3, len(df_flex))):
        print(f"\n  Sample {i+1}:")
        print(f"    UUID: {df_flex.iloc[i]['uuid']}")
        print(f"    Answer: {df_flex.iloc[i]['answers']}")
        print(f"    Prediction: {df_flex.iloc[i]['prediction']}")
        print(f"    Generation: {df_flex.iloc[i]['generation'][:100]}...")
    
    # Compare with ensemble methods if available
    print(f"\n{'='*70}")
    print("Comparison with Traditional Ensemble Methods")
    print(f"{'='*70}")
    
    comparison_results = []
    
    for method in ["avg", "max", "weighted_avg", "weighted_max"]:
        ensemble_file = os.path.join(dataset_root, f"ensemble_{method}-{num_paraphrases}.feather")
        
        if os.path.exists(ensemble_file):
            df_ensemble = pd.read_feather(ensemble_file)
            
            if "predict_lemma" in df_ensemble.columns and "answer_lemmas" in df_ensemble.columns:
                df_ensemble["answer_lemmas"] = df_ensemble["answer_lemmas"].apply(
                    lambda xs: [list(x) for x in xs] if isinstance(xs, list) else xs
                )
                answers = df_ensemble["answer_lemmas"].tolist()
                predictions = df_ensemble['predict_lemma'].tolist()
                
                acc = partial_match_scores(predictions, answers)
                comparison_results.append((method, acc))
                print(f"  {method:15s}: {acc:.3f}")
        else:
            print(f"  {method:15s}: (not available)")
    
    if flex_acc is not None and comparison_results:
        print(f"\n  {'FlexAttention':15s}: {flex_acc:.3f}")
        print()
        
        # Compute improvements
        avg_traditional = sum(acc for _, acc in comparison_results) / len(comparison_results)
        improvement = flex_acc - avg_traditional
        
        print(f"\n📊 Summary:")
        print(f"   Average traditional ensemble: {avg_traditional:.3f}")
        print(f"   FlexAttention: {flex_acc:.3f}")
        print(f"   Improvement: {improvement:+.3f} ({improvement/avg_traditional*100:+.1f}%)")
    
    print(f"\n{'='*70}")
    print("Analysis Complete")
    print(f"{'='*70}")


def compare_different_num_paraphrases(dataset_root, max_paraphrases=10):
    """
    Compare FlexAttention performance with different numbers of paraphrases.
    
    Args:
        dataset_root: Path to dataset results directory
        max_paraphrases: Maximum number of paraphrases to check
    """
    
    print("\n" + "="*70)
    print("FlexAttention: Effect of Number of Paraphrases")
    print("="*70)
    
    results = []
    
    for n in range(2, max_paraphrases + 1):
        flex_file = os.path.join(dataset_root, f"flex_attention-{n}.feather")
        
        if os.path.exists(flex_file):
            df = pd.read_feather(flex_file)
            
            if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
                df["answer_lemmas"] = df["answer_lemmas"].apply(
                    lambda xs: [list(x) for x in xs] if isinstance(xs, list) else xs
                )
                answers = df["answer_lemmas"].tolist()
                predictions = df['predict_lemma'].tolist()
                
                acc = partial_match_scores(predictions, answers)
                results.append((n, acc))
                print(f"  {n} paraphrases: {acc:.3f}")
    
    if results:
        print("\n📊 Best configuration:")
        best_n, best_acc = max(results, key=lambda x: x[1])
        print(f"   {best_n} paraphrases with accuracy {best_acc:.3f}")
    else:
        print("\n⚠️  No FlexAttention results found for comparison")
        print("   Generate results with different --num_paraphrases values")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FlexAttention generation results"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["webqa", "myriadlama"],
        help="Dataset name"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g., qwen2.5_7b_it, llama3.2_3b_it)"
    )
    parser.add_argument(
        "--num_paraphrases", type=int, default=5,
        help="Number of paraphrases to analyze (default: 5)"
    )
    parser.add_argument(
        "--compare_all", action="store_true",
        help="Compare results with different numbers of paraphrases"
    )
    
    args = parser.parse_args()
    
    # Construct dataset root path
    dataset_root = f"datasets/{args.dataset}/{args.model}"
    
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset root not found: {dataset_root}")
        print(f"   Please check the dataset and model names")
        return
    
    # Run analysis
    analyze_flexattention_results(dataset_root, args.num_paraphrases)
    
    # Compare different numbers of paraphrases if requested
    if args.compare_all:
        compare_different_num_paraphrases(dataset_root)


if __name__ == "__main__":
    main()
