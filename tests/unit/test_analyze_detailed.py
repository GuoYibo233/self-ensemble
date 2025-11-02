#!/usr/bin/env python3
"""
Test script for analyze_detailed.py

Creates mock data and tests the detailed analysis functionality.
"""

import os
import sys
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.analyze_detailed import (
    prepare_detailed_table,
    calculate_accuracy,
    export_detailed_table
)


def create_mock_baseline_origin_data():
    """Create mock data that mimics baseline_origin.feather structure."""
    data = {
        "uuid": ["q1", "q2", "q3"],
        "answers": [
            ["Paris", "paris"],
            ["42", "forty-two"],
            ["Shakespeare", "William Shakespeare"]
        ],
        "question": [
            "What is the capital of France?",
            "What is the answer to life?",
            "Who wrote Hamlet?"
        ],
        "prompt": [
            "Q: What is the capital of France?\nA:",
            "Q: What is the answer to life?\nA:",
            "Q: Who wrote Hamlet?\nA:"
        ],
        "prediction": [
            "Paris",
            "42",
            "Shakespeare"
        ],
        "generation": [
            "Paris is the capital",
            "42 is the answer",
            "Shakespeare wrote it"
        ],
        "predict_lemma": [
            ["paris"],
            ["42"],
            ["shakespeare"]
        ],
        "answer_lemmas": [
            [["paris"], ["paris"]],
            [["42"], ["forty", "two"]],
            [["shakespeare"], ["william", "shakespeare"]]
        ]
    }
    return pd.DataFrame(data)


def create_mock_baseline_per_prompt_data():
    """Create mock data that mimics baseline_per_prompt.feather structure."""
    data = {
        "uuid": ["q1", "q1", "q2", "q2"],
        "answers": [
            ["Paris", "paris"],
            ["Paris", "paris"],
            ["42", "forty-two"],
            ["42", "forty-two"]
        ],
        "paraphrase": [
            "What is the capital of France?",
            "Which city is the capital of France?",
            "What is the answer to life?",
            "What is the meaning of life?"
        ],
        "prompt": [
            "Q: What is the capital of France?\nA:",
            "Q: Which city is the capital of France?\nA:",
            "Q: What is the answer to life?\nA:",
            "Q: What is the meaning of life?\nA:"
        ],
        "prediction": [
            "Paris",
            "Paris",
            "42",
            "meaning"
        ],
        "generation": [
            "Paris is the capital",
            "Paris is the capital city",
            "42 is the answer",
            "meaning is complex"
        ],
        "predict_lemma": [
            ["paris"],
            ["paris"],
            ["42"],
            ["meaning"]
        ],
        "answer_lemmas": [
            [["paris"], ["paris"]],
            [["paris"], ["paris"]],
            [["42"], ["forty", "two"]],
            [["42"], ["forty", "two"]]
        ]
    }
    return pd.DataFrame(data)


def create_mock_flex_attention_data():
    """Create mock data that mimics flex_attention.feather structure."""
    data = {
        "uuid": ["q1", "q2"],
        "answers": [
            ["Paris", "paris"],
            ["42", "forty-two"]
        ],
        "paraphrases": [
            ("What is the capital of France?", "Which city is the capital of France?", "France's capital?"),
            ("What is the answer to life?", "What is the meaning of life?", "Life's answer?")
        ],
        "prediction": [
            "Paris",
            "42"
        ],
        "generation": [
            "Paris is the capital of France",
            "42 is the answer to everything"
        ],
        "predict_lemma": [
            ["paris"],
            ["42"]
        ],
        "answer_lemmas": [
            [["paris"], ["paris"]],
            [["42"], ["forty", "two"]]
        ]
    }
    return pd.DataFrame(data)


def test_prepare_detailed_table():
    """Test prepare_detailed_table function."""
    print("Testing prepare_detailed_table...")
    
    # Test with baseline_origin data
    df = create_mock_baseline_origin_data()
    detailed_df = prepare_detailed_table(df, "baseline_origin")
    
    assert len(detailed_df) == len(df), "Detailed table should have same number of rows"
    assert "UUID" in detailed_df.columns, "Should have UUID column"
    assert "Original_Question" in detailed_df.columns, "Should have Original_Question column"
    assert "Model_Input_Prompt" in detailed_df.columns, "Should have Model_Input_Prompt column"
    assert "Processed_Output_Prediction" in detailed_df.columns, "Should have Processed_Output_Prediction column"
    assert "Is_Correct" in detailed_df.columns, "Should have Is_Correct column"
    
    # Check accuracy markers
    assert detailed_df.iloc[0]["Is_Correct"] == "✓", "First prediction should be correct"
    assert detailed_df.iloc[1]["Is_Correct"] == "✓", "Second prediction should be correct"
    
    print("✅ prepare_detailed_table test passed")


def test_calculate_accuracy():
    """Test calculate_accuracy function."""
    print("Testing calculate_accuracy...")
    
    df = create_mock_baseline_origin_data()
    accuracy = calculate_accuracy(df)
    
    assert accuracy is not None, "Should calculate accuracy"
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert accuracy == 1.0, "All predictions in mock data should be correct"
    
    print(f"✅ calculate_accuracy test passed (accuracy: {accuracy:.3f})")


def test_export_detailed_table():
    """Test export_detailed_table function."""
    print("Testing export_detailed_table...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        df = create_mock_baseline_origin_data()
        detailed_df = prepare_detailed_table(df, "baseline_origin")
        
        # Test CSV export
        output_path = os.path.join(temp_dir, "test_output")
        csv_file = export_detailed_table(detailed_df, output_path, "csv")
        assert os.path.exists(csv_file), "CSV file should be created"
        
        # Verify CSV can be read back
        df_read = pd.read_csv(csv_file)
        assert len(df_read) == len(detailed_df), "CSV should have same number of rows"
        
        print(f"✅ CSV export test passed (file: {csv_file})")
        
        # Test Excel export (only if openpyxl is available)
        try:
            import openpyxl
            output_path2 = os.path.join(temp_dir, "test_output2")
            excel_file = export_detailed_table(detailed_df, output_path2, "excel")
            assert os.path.exists(excel_file), "Excel file should be created"
            print(f"✅ Excel export test passed (file: {excel_file})")
        except ImportError:
            print("⚠️  openpyxl not available, skipping Excel export test")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_different_data_formats():
    """Test with different data formats."""
    print("Testing different data formats...")
    
    # Test baseline_per_prompt
    df = create_mock_baseline_per_prompt_data()
    detailed_df = prepare_detailed_table(df, "baseline_per_prompt")
    assert "Paraphrase" in detailed_df.columns, "Should have Paraphrase column for per_prompt data"
    print("✅ baseline_per_prompt format test passed")
    
    # Test flex_attention
    df = create_mock_flex_attention_data()
    detailed_df = prepare_detailed_table(df, "flex_attention")
    assert any("Paraphrase" in col for col in detailed_df.columns), "Should have Paraphrase columns for flex_attention data"
    print("✅ flex_attention format test passed")


def main():
    """Run all tests."""
    print("="*70)
    print("Running tests for analyze_detailed.py")
    print("="*70)
    print()
    
    try:
        test_prepare_detailed_table()
        test_calculate_accuracy()
        test_export_detailed_table()
        test_different_data_formats()
        
        print()
        print("="*70)
        print("✅ All tests passed!")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
