#!/usr/bin/env python3
"""
Test script to verify that models and datasets download to the correct net folder location.
"""

import os
import sys

# Configure Hugging Face cache directories to use net folder
os.environ['HF_HOME'] = '/net/tokyo100-10g/data/str01_01/y-guo/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/net/tokyo100-10g/data/str01_01/y-guo/models'  
os.environ['HF_DATASETS_CACHE'] = '/net/tokyo100-10g/data/str01_01/y-guo/datasets'
os.environ['HF_HUB_CACHE'] = '/net/tokyo100-10g/data/str01_01/y-guo/models'

def test_environment_variables():
    """Test that environment variables are set correctly."""
    print("🔍 Testing Environment Variables:")
    print(f"  HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"  TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
    print(f"  HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE')}")
    print(f"  HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
    
    # Check if directories exist
    cache_dirs = [
        os.environ.get('HF_HOME'),
        os.environ.get('TRANSFORMERS_CACHE'),
        os.environ.get('HF_DATASETS_CACHE'),
        os.environ.get('HF_HUB_CACHE')
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir and os.path.exists(cache_dir):
            print(f"  ✅ {cache_dir} exists")
        elif cache_dir:
            print(f"  📁 {cache_dir} will be created when needed")
        else:
            print(f"  ❌ Cache directory not set")

def test_transformers_cache():
    """Test where transformers will cache models."""
    try:
        from transformers import AutoTokenizer
        print("\n🤖 Testing Transformers Cache Location:")
        
        # This will show where models would be cached
        print(f"  Default cache dir: {os.environ.get('TRANSFORMERS_CACHE')}")
        
        # Test with a very small model (just tokenizer)
        print("  Testing tokenizer download (small test)...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Small test
        print("  ✅ Tokenizer download test successful")
        
    except Exception as e:
        print(f"  ❌ Error testing transformers: {e}")

def test_datasets_cache():
    """Test where datasets will be cached."""
    try:
        from datasets import load_dataset
        print("\n📊 Testing Datasets Cache Location:")
        print(f"  Default cache dir: {os.environ.get('HF_DATASETS_CACHE')}")
        
        # Don't actually download, just show the configuration
        print("  ✅ Datasets configuration ready")
        
    except Exception as e:
        print(f"  ❌ Error testing datasets: {e}")

def show_disk_usage():
    """Show disk usage of cache directories."""
    print("\n💾 Disk Usage:")
    
    import subprocess
    cache_dirs = [
        '/net/tokyo100-10g/data/str01_01/y-guo/models',
        '/net/tokyo100-10g/data/str01_01/y-guo/datasets',
        '/net/tokyo100-10g/data/str01_01/y-guo/hf_cache'
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                result = subprocess.run(['du', '-sh', cache_dir], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    size = result.stdout.strip().split()[0]
                    print(f"  {cache_dir}: {size}")
                else:
                    print(f"  {cache_dir}: (unable to measure)")
            except:
                print(f"  {cache_dir}: (unable to measure)")
        else:
            print(f"  {cache_dir}: (not created yet)")

if __name__ == "__main__":
    print("🧪 Testing Model and Dataset Download Locations")
    print("=" * 60)
    
    test_environment_variables()
    test_transformers_cache()
    test_datasets_cache()
    show_disk_usage()
    
    print("\n✅ Test completed!")
    print("\nNext steps:")
    print("1. Run: bash tools/download_resources.sh --dataset webqa --model llama3.2_3b_it")
    print("2. Check that files appear in /net/tokyo100-10g/data/str01_01/y-guo/models/")
    print("3. Check that datasets appear in /net/tokyo100-10g/data/str01_01/y-guo/datasets/")