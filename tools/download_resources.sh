#!/bin/bash
#
# FlexAttention Resource Download Script
#
# This script helps download necessary datasets and models for FlexAttention
# ensemble generation.
#
# Usage:
#   bash download_resources.sh --dataset webqa --model llama3.2_3b_it
#   bash download_resources.sh --dataset myriadlama --model llama3.2_3b_it
#   bash download_resources.sh --list
#   bash download_resources.sh --dataset-only webqa
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found. Please install Python 3.10+"
        exit 1
    fi
    print_info "Python found: $(python3 --version)"
}

# Function to check disk space
check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        print_warning "Low disk space: ${available_gb}GB available, ${required_gb}GB recommended"
        return 1
    else
        print_info "Disk space: ${available_gb}GB available"
        return 0
    fi
}

# Function to create Python script for downloading datasets
download_dataset() {
    local dataset_name=$1
    
    print_header "Downloading Dataset: $dataset_name"
    
    python3 << EOF
import sys
import os

# Add repository to path
sys.path.insert(0, '${PWD}')

try:
    print("Importing dataset module...")
    if "${dataset_name}" == "webqa":
        from dataset import WebQADataset
        print("Creating WebQA dataset...")
        dataset = WebQADataset(model_name="llama3.2_3b_it")
        print("✓ WebQA dataset ready")
        print(f"  Location: {dataset.dataset_root}")
        print(f"  Samples: {len(dataset.ds)}")
    elif "${dataset_name}" == "myriadlama":
        from dataset import MyriadLamaDataset
        print("Creating MyriadLAMA dataset...")
        dataset = MyriadLamaDataset(model_name="llama3.2_3b_it")
        print("✓ MyriadLAMA dataset ready")
        print(f"  Location: {dataset.dataset_root}")
        print(f"  Samples: {len(dataset.ds)}")
    else:
        print(f"✗ Unknown dataset: ${dataset_name}")
        sys.exit(1)
    
    print("\n✓ Dataset download complete!")
    
except Exception as e:
    print(f"✗ Error downloading dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_info "Dataset $dataset_name downloaded successfully"
    else
        print_error "Failed to download dataset $dataset_name"
        exit 1
    fi
}

# Function to download model
download_model() {
    local model_name=$1
    
    print_header "Downloading Model: $model_name"
    
    python3 << EOF
import sys
import os

# Add repository to path
sys.path.insert(0, '${PWD}')

try:
    from constants import MODEL_PATHs
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    if "${model_name}" not in MODEL_PATHs:
        print(f"✗ Unknown model: ${model_name}")
        print(f"  Available models: {list(MODEL_PATHs.keys())}")
        sys.exit(1)
    
    model_path = MODEL_PATHs["${model_name}"]
    print(f"Model path: {model_path}")
    
    # Check cache first
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"Checking cache: {cache_dir}")
    
    print("\nDownloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✓ Tokenizer downloaded")
    
    print("\nDownloading model (this may take a while)...")
    print("  Note: Model will be downloaded to ~/.cache/huggingface/")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",  # Use CPU to avoid GPU memory during download
        torch_dtype="auto"
    )
    print("✓ Model downloaded")
    
    # Get model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model ready:")
    print(f"  Parameters: {param_count:,}")
    print(f"  Size: ~{param_count * 2 / 1e9:.1f}GB (fp16)")
    
except Exception as e:
    print(f"✗ Error downloading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_info "Model $model_name downloaded successfully"
    else
        print_error "Failed to download model $model_name"
        exit 1
    fi
}

# Function to download spaCy model
download_spacy_model() {
    print_header "Downloading spaCy Model"
    
    python3 -m spacy download en_core_web_lg
    
    if [ $? -eq 0 ]; then
        print_info "spaCy model downloaded successfully"
    else
        print_error "Failed to download spaCy model"
        exit 1
    fi
}

# Function to list available resources
list_resources() {
    print_header "Available Resources"
    
    python3 << EOF
import sys
sys.path.insert(0, '${PWD}')

try:
    from constants import MODEL_PATHs
    
    print("\n📦 Available Models:")
    for name, path in MODEL_PATHs.items():
        print(f"  - {name}")
        print(f"    Path: {path}")
    
    print("\n📊 Available Datasets:")
    print("  - webqa")
    print("    Description: WebQA question answering dataset")
    print("  - myriadlama")
    print("    Description: MyriadLAMA knowledge probing dataset")
    
    print("\n📚 Required spaCy Model:")
    print("  - en_core_web_lg")
    print("    Description: English language model for lemmatization")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
EOF
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: bash download_resources.sh [OPTIONS]

Options:
    --dataset DATASET       Download specific dataset (webqa, myriadlama)
    --model MODEL          Download specific model (from constants.MODEL_PATHs)
    --dataset-only DATASET Download only dataset, skip model
    --spacy                Download only spaCy model
    --list                 List available resources
    --help                 Show this help message

Examples:
    # Download WebQA dataset and Llama model
    bash download_resources.sh --dataset webqa --model llama3.2_3b_it
    
    # Download only MyriadLAMA dataset
    bash download_resources.sh --dataset-only myriadlama
    
    # Download only spaCy model
    bash download_resources.sh --spacy
    
    # List available resources
    bash download_resources.sh --list

EOF
}

# Main script
main() {
    local dataset=""
    local model=""
    local dataset_only=false
    local spacy_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset)
                dataset="$2"
                shift 2
                ;;
            --model)
                model="$2"
                shift 2
                ;;
            --dataset-only)
                dataset="$2"
                dataset_only=true
                shift 2
                ;;
            --spacy)
                spacy_only=true
                shift
                ;;
            --list)
                check_python
                list_resources
                exit 0
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check if no arguments provided
    if [ -z "$dataset" ] && [ -z "$model" ] && [ "$spacy_only" = false ]; then
        show_usage
        exit 1
    fi
    
    # Check prerequisites
    check_python
    check_disk_space 20  # Warn if less than 20GB available
    
    print_header "FlexAttention Resource Downloader"
    
    # Download spaCy model if requested
    if [ "$spacy_only" = true ]; then
        download_spacy_model
        exit 0
    fi
    
    # Download dataset
    if [ -n "$dataset" ]; then
        download_dataset "$dataset"
    fi
    
    # Download model (unless dataset-only)
    if [ -n "$model" ] && [ "$dataset_only" = false ]; then
        download_model "$model"
    fi
    
    # Download spaCy model
    download_spacy_model
    
    print_header "Download Complete!"
    print_info "All resources have been downloaded successfully"
    
    echo ""
    echo "Next steps:"
    echo "  1. Validate environment: python3 validate_flexattention_env.py"
    echo "  2. Run debug script: python3 debug_flexattention.py --dataset $dataset --model ${model:-llama3.2_3b_it} --max-samples 1"
    echo "  3. Run generation: python3 flex_attention_generate.py --dataset $dataset --model ${model:-llama3.2_3b_it}"
}

# Run main function
main "$@"
