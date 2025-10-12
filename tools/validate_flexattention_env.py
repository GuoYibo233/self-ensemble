#!/usr/bin/env python3
"""
FlexAttention Environment Validation Script

This script validates that your environment is properly configured for running
the FlexAttention-based ensemble generation code.

Usage:
    python3 validate_flexattention_env.py
    python3 validate_flexattention_env.py --verbose
    python3 validate_flexattention_env.py --test-flex-attention
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


class EnvironmentValidator:
    """Validates the environment for FlexAttention code."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.passed = []
    
    def log(self, message, level="INFO"):
        """Log a message with optional verbosity control."""
        if level == "ERROR":
            self.errors.append(message)
            print(f"❌ {message}")
        elif level == "WARNING":
            self.warnings.append(message)
            print(f"⚠️  {message}")
        elif level == "PASS":
            self.passed.append(message)
            print(f"✅ {message}")
        else:
            if self.verbose:
                print(f"ℹ️  {message}")
    
    def check_python_version(self):
        """Check Python version is 3.10 or higher."""
        print("\n🔍 Checking Python version...")
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major >= 3 and version.minor >= 10:
            self.log(f"Python {version_str} - Compatible", "PASS")
            return True
        else:
            self.log(f"Python {version_str} - Requires Python 3.10+", "ERROR")
            return False
    
    def check_pytorch(self):
        """Check PyTorch installation and version."""
        print("\n🔍 Checking PyTorch...")
        try:
            import torch
            version = torch.__version__
            self.log(f"PyTorch {version} - Installed", "PASS")
            
            # Check CUDA
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
                self.log(f"CUDA {cuda_version} available - {device_count} device(s)", "PASS")
                self.log(f"GPU: {device_name}", "INFO")
            else:
                self.log("CUDA not available - will use CPU (slower)", "WARNING")
            
            return True
        except ImportError:
            self.log("PyTorch not installed", "ERROR")
            self.log("Install with: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121", "INFO")
            return False
    
    def check_flex_attention(self):
        """Check if FlexAttention is available."""
        print("\n🔍 Checking FlexAttention API...")
        try:
            from torch.nn.attention.flex_attention import flex_attention, create_block_mask
            self.log("FlexAttention API available", "PASS")
            return True
        except ImportError:
            self.log("FlexAttention not available - requires PyTorch 2.5+ or nightly", "ERROR")
            self.log("Install with: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121", "INFO")
            return False
    
    def check_dependencies(self):
        """Check required Python packages."""
        print("\n🔍 Checking dependencies...")
        required = {
            'transformers': 'transformers',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'tqdm': 'tqdm',
            'datasets': 'datasets',
            'spacy': 'spacy',
        }
        
        all_installed = True
        for package, import_name in required.items():
            try:
                mod = __import__(import_name)
                version = getattr(mod, '__version__', 'unknown')
                self.log(f"{package} {version} - Installed", "PASS")
            except ImportError:
                self.log(f"{package} not installed", "ERROR")
                all_installed = False
        
        return all_installed
    
    def check_spacy_model(self):
        """Check if spaCy language model is downloaded."""
        print("\n🔍 Checking spaCy model...")
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_lg")
                self.log("spaCy model 'en_core_web_lg' - Available", "PASS")
                return True
            except OSError:
                self.log("spaCy model 'en_core_web_lg' not found", "WARNING")
                self.log("Download with: python3 -m spacy download en_core_web_lg", "INFO")
                return False
        except ImportError:
            self.log("spaCy not installed", "ERROR")
            return False
    
    def check_disk_space(self):
        """Check available disk space."""
        print("\n🔍 Checking disk space...")
        try:
            stat = os.statvfs('.')
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            
            if available_gb > 20:
                self.log(f"Disk space: {available_gb:.1f} GB available - Sufficient", "PASS")
                return True
            elif available_gb > 10:
                self.log(f"Disk space: {available_gb:.1f} GB available - May be tight for large models", "WARNING")
                return True
            else:
                self.log(f"Disk space: {available_gb:.1f} GB available - Insufficient (need 20GB+)", "ERROR")
                return False
        except Exception as e:
            self.log(f"Could not check disk space: {e}", "WARNING")
            return True
    
    def check_memory(self):
        """Check available RAM."""
        print("\n🔍 Checking system memory...")
        try:
            # Try to read /proc/meminfo on Linux
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemAvailable' in line:
                            available_kb = int(line.split()[1])
                            available_gb = available_kb / (1024**2)
                            
                            if available_gb > 16:
                                self.log(f"RAM: {available_gb:.1f} GB available - Sufficient", "PASS")
                                return True
                            elif available_gb > 8:
                                self.log(f"RAM: {available_gb:.1f} GB available - May be tight", "WARNING")
                                return True
                            else:
                                self.log(f"RAM: {available_gb:.1f} GB available - Insufficient (need 16GB+)", "WARNING")
                                return True
            else:
                self.log("Could not determine available memory", "INFO")
                return True
        except Exception as e:
            self.log(f"Could not check memory: {e}", "INFO")
            return True
    
    def test_flex_attention_basic(self):
        """Run a basic FlexAttention test."""
        print("\n🔍 Testing FlexAttention functionality...")
        try:
            import torch
            from torch.nn.attention.flex_attention import flex_attention, create_block_mask
            
            # Create small test tensors
            B, H, S, D = 1, 2, 8, 16
            Q = torch.randn(B, H, S, D, device='cuda' if torch.cuda.is_available() else 'cpu')
            K = torch.randn(B, H, S, D, device='cuda' if torch.cuda.is_available() else 'cpu')
            V = torch.randn(B, H, S, D, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create a simple causal mask
            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
            
            # Create block mask
            block_mask = create_block_mask(
                causal_mask, B=B, H=H, Q_LEN=S, KV_LEN=S,
                device=Q.device
            )
            
            # Run FlexAttention
            output = flex_attention(Q, K, V, block_mask=block_mask)
            
            # Verify output shape
            if output.shape == (B, H, S, D):
                self.log("FlexAttention test passed - Working correctly", "PASS")
                return True
            else:
                self.log(f"FlexAttention test failed - Unexpected output shape: {output.shape}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"FlexAttention test failed: {e}", "ERROR")
            return False
    
    def check_repository_files(self):
        """Check that required repository files exist."""
        print("\n🔍 Checking repository files...")
        required_files = [
            'flex_attention_generate.py',
            'dataset.py',
            'constants.py',
            'generate.py',
        ]
        
        all_exist = True
        for file in required_files:
            if os.path.exists(file):
                self.log(f"{file} - Found", "PASS")
            else:
                self.log(f"{file} - Missing", "ERROR")
                all_exist = False
        
        return all_exist
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        print(f"✅ Passed: {len(self.passed)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        print("="*60)
        
        if self.errors:
            print("\n❌ ERRORS FOUND:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors:
            print("\n🎉 Environment validation PASSED!")
            print("   You're ready to run FlexAttention code.")
            return True
        else:
            print("\n❌ Environment validation FAILED!")
            print("   Please fix the errors above before proceeding.")
            return False
    
    def run_all_checks(self, test_flex_attention=False):
        """Run all validation checks."""
        print("="*60)
        print("🔧 FlexAttention Environment Validation")
        print("="*60)
        
        self.check_python_version()
        self.check_pytorch()
        self.check_flex_attention()
        self.check_dependencies()
        self.check_spacy_model()
        self.check_disk_space()
        self.check_memory()
        self.check_repository_files()
        
        if test_flex_attention:
            self.test_flex_attention_basic()
        
        return self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Validate environment for FlexAttention code"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--test-flex-attention',
        action='store_true',
        help='Run FlexAttention functionality test'
    )
    args = parser.parse_args()
    
    validator = EnvironmentValidator(verbose=args.verbose)
    success = validator.run_all_checks(test_flex_attention=args.test_flex_attention)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
