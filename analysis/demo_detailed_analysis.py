#!/usr/bin/env python3
"""
Demo script to show analyze_detailed.py functionality.

This creates sample data and runs the analysis to demonstrate the output.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_demo_data():
    """Create demo data in a temporary directory."""
    demo_dir = "datasets/demo/demo_model"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Create baseline_origin demo data
    data = {
        "uuid": ["q1", "q2", "q3", "q4", "q5"],
        "answers": [
            ["Paris", "paris"],
            ["42", "forty-two"],
            ["Shakespeare", "William Shakespeare"],
            ["H2O", "water"],
            ["Einstein", "Albert Einstein"]
        ],
        "question": [
            "What is the capital of France?",
            "What is the answer to life, the universe, and everything?",
            "Who wrote Hamlet?",
            "What is the chemical formula for water?",
            "Who developed the theory of relativity?"
        ],
        "prompt": [
            "Q: What is the capital of France?\nA:",
            "Q: What is the answer to life, the universe, and everything?\nA:",
            "Q: Who wrote Hamlet?\nA:",
            "Q: What is the chemical formula for water?\nA:",
            "Q: Who developed the theory of relativity?\nA:"
        ],
        "prediction": [
            "Paris",
            "42",
            "Shakespeare",
            "H2O",
            "Newton"  # Intentionally incorrect
        ],
        "generation": [
            "Paris is the capital city of France.",
            "42 is the answer to everything.",
            "Shakespeare wrote Hamlet, a famous tragedy.",
            "H2O is the chemical formula for water.",
            "Newton discovered gravity."  # Intentionally incorrect
        ],
        "predict_lemma": [
            ["paris"],
            ["42"],
            ["shakespeare"],
            ["h2o"],
            ["newton"]
        ],
        "answer_lemmas": [
            [["paris"], ["paris"]],
            [["42"], ["forty", "two"]],
            [["shakespeare"], ["william", "shakespeare"]],
            [["h2o"], ["water"]],
            [["einstein"], ["albert", "einstein"]]
        ]
    }
    df = pd.DataFrame(data)
    
    # Save to feather file
    output_file = os.path.join(demo_dir, "baseline_origin.feather")
    df.to_feather(output_file)
    print(f"✅ Demo data created: {output_file}")
    print(f"   Total samples: {len(df)}")
    print(f"   Correct answers: 4/5 (80%)")
    print()
    
    return demo_dir


def run_demo():
    """Run the demo analysis."""
    print("="*70)
    print("Demo: Enhanced Analysis Script")
    print("="*70)
    print()
    
    # Create demo data
    demo_dir = create_demo_data()
    
    # Run the analysis
    print("Running analysis on demo data...")
    print()
    
    import subprocess
    cmd = [
        "python3", "analysis/analyze_detailed.py",
        "--dataset", "demo",
        "--model", "demo_model",
        "--method", "baseline_origin",
        "--export-format", "csv"
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print()
        print("="*70)
        print("Demo completed successfully!")
        print("="*70)
        print()
        print("Check the output files:")
        print(f"  - {demo_dir}/baseline_origin_detailed.csv")
        print()
        print("You can view the CSV file to see all detailed features.")
    else:
        print("❌ Demo failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
