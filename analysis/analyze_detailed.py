#!/usr/bin/env python3
"""
Enhanced analysis script for detailed feature export.

This script analyzes generation results and exports all features to a table
for easy viewing, including:
- Original question
- Processed/paraphrased questions
- Model input (prompt)
- Model output (generation)
- Processed output (prediction)
- Correct answer
- Accuracy calculation

Usage:
    python analysis/analyze_detailed.py --dataset webqa --model llama3.2_3b_it --method baseline_origin
    python analysis/analyze_detailed.py --dataset webqa --model llama3.2_3b_it --method flex_attention --num_paraphrases 5
    python analysis/analyze_detailed.py --dataset myriadlama --model qwen2.5_7b_it --method baseline_per_prompt --export-format excel
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import partial_match, partial_match_scores


def load_results(dataset_root, method, num_paraphrases=None):
    """
    Load results from feather file based on method.
    
    Args:
        dataset_root: Path to dataset results directory
        method: Method name (baseline_origin, baseline_per_prompt, flex_attention, ensemble_*)
        num_paraphrases: Number of paraphrases (for flex_attention and ensemble methods)
    
    Returns:
        DataFrame with results
    """
    if method == "baseline_origin":
        result_file = os.path.join(dataset_root, "baseline_origin.feather")
    elif method == "baseline_per_prompt":
        result_file = os.path.join(dataset_root, "baseline_per_prompt.feather")
    elif method.startswith("flex_attention"):
        if num_paraphrases is None:
            num_paraphrases = 5
        result_file = os.path.join(dataset_root, f"flex_attention-{num_paraphrases}.feather")
    elif method.startswith("ensemble_"):
        if num_paraphrases is None:
            num_paraphrases = 5
        ensemble_type = method.replace("ensemble_", "")
        result_file = os.path.join(dataset_root, f"ensemble_{ensemble_type}-{num_paraphrases}.feather")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Results file not found: {result_file}")
    
    print(f"✅ Loading results from: {result_file}")
    df = pd.read_feather(result_file)
    return df, result_file


def prepare_detailed_table(df, method):
    """
    Prepare a detailed table with all feature information.
    
    Args:
        df: DataFrame with results
        method: Method name
    
    Returns:
        DataFrame with detailed features
    """
    detailed_data = []
    
    for idx, row in df.iterrows():
        item = {
            "Index": idx,
            "UUID": row.get("uuid", "N/A"),
            "Original_Question": row.get("question", "N/A"),
            "Model_Input_Prompt": row.get("prompt", "N/A"),
            "Model_Output_Generation": row.get("generation", "N/A"),
            "Processed_Output_Prediction": row.get("prediction", "N/A"),
            "Correct_Answers": str(row.get("answers", "N/A")),
        }
        
        # Add paraphrase information if available
        if "paraphrase" in row and pd.notna(row["paraphrase"]):
            item["Paraphrase"] = row["paraphrase"]
        elif "paraphrases" in row and pd.notna(row["paraphrases"]):
            # For flex_attention, paraphrases is a tuple/list
            paraphrases = row["paraphrases"]
            if isinstance(paraphrases, (list, tuple)):
                for i, para in enumerate(paraphrases):
                    item[f"Paraphrase_{i}"] = para
            else:
                item["Paraphrases"] = str(paraphrases)
        
        # Add lemmatized versions if available
        if "predict_lemma" in df.columns:
            predict_lemma = row.get("predict_lemma")
            if predict_lemma is not None and not (isinstance(predict_lemma, float) and pd.isna(predict_lemma)):
                item["Prediction_Lemma"] = str(predict_lemma)
        
        if "answer_lemmas" in df.columns:
            answer_lemmas = row.get("answer_lemmas")
            if answer_lemmas is not None and not (isinstance(answer_lemmas, float) and pd.isna(answer_lemmas)):
                item["Answer_Lemmas"] = str(answer_lemmas)
        
        # Calculate match for this specific prediction
        if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
            predict_lemma = row.get("predict_lemma")
            answer_lemmas = row.get("answer_lemmas")
            
            if predict_lemma is not None and answer_lemmas is not None:
                if not (isinstance(predict_lemma, float) and pd.isna(predict_lemma)):
                    if not (isinstance(answer_lemmas, float) and pd.isna(answer_lemmas)):
                        # Try to match
                        try:
                            # Handle numpy array conversion
                            if hasattr(answer_lemmas, '__iter__'):
                                answer_lemmas_list = []
                                for ans in answer_lemmas:
                                    if isinstance(ans, (list, tuple)):
                                        answer_lemmas_list.append(list(ans))
                                    else:
                                        # Handle numpy array or single item
                                        answer_lemmas_list.append(list(ans) if hasattr(ans, '__iter__') and not isinstance(ans, str) else [ans])
                                
                                is_correct = partial_match(predict_lemma, answer_lemmas_list)
                                item["Is_Correct"] = "✓" if is_correct else "✗"
                            else:
                                item["Is_Correct"] = "N/A"
                        except Exception as e:
                            # If matching fails, mark as N/A
                            item["Is_Correct"] = "N/A"
                    else:
                        item["Is_Correct"] = "N/A"
                else:
                    item["Is_Correct"] = "N/A"
            else:
                item["Is_Correct"] = "N/A"
        else:
            item["Is_Correct"] = "N/A"
        
        detailed_data.append(item)
    
    return pd.DataFrame(detailed_data)


def calculate_accuracy(df):
    """
    Calculate accuracy from DataFrame.
    
    Args:
        df: DataFrame with results
    
    Returns:
        accuracy: Float between 0 and 1
    """
    if "predict_lemma" not in df.columns or "answer_lemmas" not in df.columns:
        print("⚠️  Lemmatized results not available. Cannot calculate accuracy.")
        return None
    
    # Process lemmas if needed
    df_copy = df.copy()
    df_copy["answer_lemmas"] = df_copy["answer_lemmas"].apply(
        lambda xs: [list(x) if not isinstance(x, list) else x for x in xs] if isinstance(xs, list) else xs
    )
    
    answers = df_copy["answer_lemmas"].tolist()
    predictions = df_copy["predict_lemma"].tolist()
    
    accuracy = partial_match_scores(predictions, answers)
    return accuracy


def export_detailed_table(detailed_df, output_path, export_format="csv"):
    """
    Export detailed table to file.
    
    Args:
        detailed_df: DataFrame with detailed features
        output_path: Base path for output file
        export_format: Format for export (csv, excel)
    """
    if export_format == "csv":
        output_file = output_path + ".csv"
        detailed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ Detailed table exported to: {output_file}")
    elif export_format == "excel":
        output_file = output_path + ".xlsx"
        detailed_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✅ Detailed table exported to: {output_file}")
    else:
        raise ValueError(f"Unknown export format: {export_format}")
    
    return output_file


def display_summary_statistics(df, detailed_df, accuracy):
    """
    Display summary statistics about the results.
    
    Args:
        df: Original DataFrame
        detailed_df: Detailed DataFrame
        accuracy: Calculated accuracy
    """
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    
    print(f"Total samples: {len(df)}")
    print(f"Unique questions (UUIDs): {df['uuid'].nunique()}")
    
    if accuracy is not None:
        print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Count correct/incorrect if available
        if "Is_Correct" in detailed_df.columns:
            correct_count = (detailed_df["Is_Correct"] == "✓").sum()
            incorrect_count = (detailed_df["Is_Correct"] == "✗").sum()
            na_count = (detailed_df["Is_Correct"] == "N/A").sum()
            
            if correct_count > 0 or incorrect_count > 0:
                print(f"Correct predictions: {correct_count}")
                print(f"Incorrect predictions: {incorrect_count}")
                if na_count > 0:
                    print(f"Could not determine: {na_count}")
    
    print("\n" + "="*70)


def display_sample_data(detailed_df, num_samples=3):
    """
    Display sample data from the detailed table.
    
    Args:
        detailed_df: Detailed DataFrame
        num_samples: Number of samples to display
    """
    print("\n" + "="*70)
    print(f"Sample Data (first {num_samples} entries)")
    print("="*70)
    
    for i in range(min(num_samples, len(detailed_df))):
        row = detailed_df.iloc[i]
        print(f"\n--- Sample {i+1} ---")
        for col in detailed_df.columns:
            value = row[col]
            # Truncate long strings for display
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"{col}: {value}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze generation results with detailed feature export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze baseline origin results
  python analysis/analyze_detailed.py --dataset webqa --model llama3.2_3b_it --method baseline_origin
  
  # Analyze FlexAttention results with 5 paraphrases
  python analysis/analyze_detailed.py --dataset webqa --model llama3.2_3b_it --method flex_attention --num_paraphrases 5
  
  # Export to Excel format
  python analysis/analyze_detailed.py --dataset myriadlama --model qwen2.5_7b_it --method baseline_per_prompt --export-format excel
        """
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name (e.g., webqa, myriadlama)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g., llama3.2_3b_it, qwen2.5_7b_it)"
    )
    parser.add_argument(
        "--method", type=str, required=True,
        help="Method name (baseline_origin, baseline_per_prompt, flex_attention, ensemble_avg, etc.)"
    )
    parser.add_argument(
        "--num_paraphrases", type=int, default=None,
        help="Number of paraphrases (required for flex_attention and ensemble methods)"
    )
    parser.add_argument(
        "--export-format", type=str, default="csv",
        choices=["csv", "excel"],
        help="Export format for detailed table (default: csv)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for exported files (default: same as dataset root)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't display sample data (only export)"
    )
    
    args = parser.parse_args()
    
    # Construct dataset root path
    dataset_root = f"datasets/{args.dataset}/{args.model}"
    
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset root not found: {dataset_root}")
        print(f"   Please check the dataset and model names")
        return
    
    print("="*70)
    print("Detailed Analysis and Export")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    if args.num_paraphrases:
        print(f"Number of paraphrases: {args.num_paraphrases}")
    print()
    
    # Load results
    try:
        df, result_file = load_results(dataset_root, args.method, args.num_paraphrases)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    # Calculate accuracy
    accuracy = calculate_accuracy(df)
    
    # Prepare detailed table
    print("\n📊 Preparing detailed feature table...")
    detailed_df = prepare_detailed_table(df, args.method)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = dataset_root
    
    # Construct output filename
    if args.num_paraphrases:
        output_base = os.path.join(output_dir, f"{args.method}-{args.num_paraphrases}_detailed")
    else:
        output_base = os.path.join(output_dir, f"{args.method}_detailed")
    
    # Export detailed table
    print(f"\n📤 Exporting detailed table...")
    output_file = export_detailed_table(detailed_df, output_base, args.export_format)
    
    # Display summary statistics
    display_summary_statistics(df, detailed_df, accuracy)
    
    # Display sample data if not disabled
    if not args.no_display:
        display_sample_data(detailed_df, num_samples=3)
    
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
    print(f"Detailed table exported to: {output_file}")
    print(f"Total entries: {len(detailed_df)}")
    if accuracy is not None:
        print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
