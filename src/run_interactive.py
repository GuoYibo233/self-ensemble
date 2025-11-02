#!/usr/bin/env python3
"""
Interactive generation script with parameter prompts.

This script provides an interactive interface for running generation experiments.
It prompts for missing parameters and provides selection menus for models and datasets.
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.interactive import prompt_for_parameter, select_from_options, confirm_action
from core.constants import MODEL_PATHs


def get_available_models():
    """Get list of available models from constants."""
    return list(MODEL_PATHs.keys())


def get_available_datasets():
    """Get list of available datasets."""
    return ['webqa', 'myriadlama']


def get_available_methods(generation_type):
    """Get list of available methods based on generation type."""
    if generation_type == 'original':
        return ['avg', 'max', 'weighted_avg', 'weighted_max', 'per_prompt']
    elif generation_type == 'baseline':
        return ['origin', 'per_prompt', 'all']
    else:
        return None  # FlexAttention and MyriadLAMA don't have method selection


def main():
    """Main interactive generation interface."""
    
    print("=" * 80)
    print("Self-Ensemble Generation - Interactive Mode")
    print("=" * 80)
    
    # Select generation type
    generation_types = [
        'original',          # Original ensemble methods (generate.py)
        'flex_attention',    # FlexAttention-based (flex_attention_generate.py)
        'myriadlama',        # MyriadLAMA-specific (myriadlama_flex_attention_generate.py)
        'baseline'           # Baseline generation (baseline_generate.py)
    ]
    
    gen_type = select_from_options(
        'generation_type',
        'Select generation type',
        generation_types,
        default='flex_attention'
    )
    
    # Select dataset
    dataset = select_from_options(
        'dataset',
        'Select dataset',
        get_available_datasets(),
        default='webqa'
    )
    
    # Select model
    model = select_from_options(
        'model',
        'Select model',
        get_available_models(),
        default='llama3.2_3b_it'
    )
    
    # Method selection (only for original and baseline)
    method = None
    if gen_type in ['original', 'baseline']:
        methods = get_available_methods(gen_type)
        method = select_from_options(
            'method',
            'Select generation method',
            methods,
            default=methods[0]
        )
    
    # Number of paraphrases (for flex_attention and myriadlama)
    num_paraphrases = None
    if gen_type in ['flex_attention', 'myriadlama']:
        num_paraphrases_str = prompt_for_parameter(
            'num_paraphrases',
            'Number of paraphrases',
            default='5'
        )
        num_paraphrases = int(num_paraphrases_str)
    
    # Number of ensemble members (for original)
    num_ensemble = None
    if gen_type == 'original' and method in ['avg', 'max', 'weighted_avg', 'weighted_max']:
        num_ensemble_str = prompt_for_parameter(
            'num_ensemble',
            'Number of ensemble members',
            default='6'
        )
        num_ensemble = int(num_ensemble_str)
    
    # Max samples (optional)
    if confirm_action('Limit number of samples?', default=False):
        max_samples_str = prompt_for_parameter(
            'max_samples',
            'Maximum number of samples',
            default='100'
        )
        max_samples = int(max_samples_str)
    else:
        max_samples = None
    
    # Device selection
    device = select_from_options(
        'device',
        'Select device',
        ['cuda', 'cpu'],
        default='cuda'
    )
    
    # Build command
    script_map = {
        'original': 'generate_original.py',
        'flex_attention': 'generate_flex_attention.py',
        'myriadlama': 'generate_myriadlama.py',
        'baseline': 'generate_baseline.py'
    }
    
    script_path = os.path.join(os.path.dirname(__file__), script_map[gen_type])
    cmd_args = [
        sys.executable,
        script_path,
        '--dataset', dataset,
        '--model', model,
        '--device', device
    ]
    
    if method:
        cmd_args.extend(['--method', method])
    
    if num_paraphrases:
        cmd_args.extend(['--num_paraphrases', str(num_paraphrases)])
    
    if num_ensemble:
        cmd_args.extend(['--num_ensemble', str(num_ensemble)])
    
    if max_samples:
        cmd_args.extend(['--max_samples', str(max_samples)])
    
    # Show command and confirm
    print("\n" + "=" * 80)
    print("Command to be executed:")
    print(" ".join(cmd_args))
    print("=" * 80)
    
    if confirm_action('Execute this command?', default=True):
        import subprocess
        result = subprocess.run(cmd_args)
        sys.exit(result.returncode)
    else:
        print("Cancelled.")
        sys.exit(0)


if __name__ == '__main__':
    main()
