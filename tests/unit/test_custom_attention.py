#!/usr/bin/env python3
"""
Unit tests for myriadlama_custom_attention_generate.py

Tests the core functionality:
1. build_question_struct_mask - Custom attention mask building
2. tokenize_prompt_parts - Tokenization with position tracking
3. Mask behavior verification

Note: Core functions are copied here rather than imported to allow isolated testing
without requiring all module dependencies (spacy, transformers, etc.). This is a
trade-off: the duplicated code may drift from the main module, but it enables
running these tests in minimal CI environments. The functions tested here are the
core algorithms that should not change frequently.
"""

import pytest
import torch
import sys
import os


# Copy the core functions here for isolated testing without full module dependencies.
# This enables testing in environments without spacy, transformers, etc. installed.
# If these functions change in the main module, update them here as well.
def build_question_struct_mask(question_group_ids, Q, K, dtype, device):
    """
    Build the per-query structure mask [1, 1, Q, K].
    
    Rule (token-level):
    - Let g_q = question_group_ids[q], g_k = question_group_ids[k].
    - If g_q == 0 (shared part): allow everything (structure doesn't constrain it)
    - If g_q >= 1 (para part N): allow g_k == 0 OR g_k == g_q (can see shared + self)
    
    Args:
        question_group_ids: [S] tensor with values in {0, 1, 2, ...}
                           0 = shared part, 1+ = para parts
        Q: Query sequence length
        K: Key sequence length
        dtype: Data type for mask
        device: Device for mask
        
    Returns:
        [1, 1, Q, K] additive mask (0 or -inf)
    """
    neg_inf = torch.finfo(dtype).min
    
    # Ensure question_group_ids is on the correct device
    q_group_ids = question_group_ids[:Q].to(device)
    k_group_ids = question_group_ids[:K].to(device)
    
    # [Q, 1] and [1, K]
    g_q = q_group_ids.view(Q, 1)  # [Q, 1]
    g_k = k_group_ids.view(1, K)  # [1, K]
    
    # Boolean conditions
    is_shared = (g_q == 0)  # Query is in shared part
    is_para = (g_q >= 1)    # Query is in some para part
    
    # For shared part queries: allow all keys (normal causal will be applied separately)
    allowed_shared = is_shared  # [Q, 1] broadcast to [Q, K]
    
    # For para part queries: allow keys in shared (g_k == 0) OR same group (g_q == g_k)
    is_k_shared = (g_k == 0)  # [1, K]
    same_group = (g_q == g_k)  # [Q, K]
    allowed_para = is_para & (is_k_shared | same_group)  # [Q, K]
    
    # Combine: allow if either condition is true
    allowed = allowed_shared | allowed_para  # [Q, K]
    
    # Create mask: 0 for allowed, -inf for blocked
    mask = torch.zeros((1, 1, Q, K), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, neg_inf)
    
    return mask


def construct_prompt_new_format(instruction, few_shot_examples, question_paraphrases):
    """
    Construct prompt parts: one shared part + multiple paraphrase parts.
    """
    # Build shared part: instruction + few-shot examples
    shared_parts = []
    
    # Add instruction with trailing newline
    shared_parts.append(instruction + "\n")
    
    # Add few-shot examples (each on one line: Q: ... A: ...)
    fs_lines = []
    for fs_example in few_shot_examples:
        # Use only the FIRST paraphrase from each few-shot example
        para = fs_example['paraphrases'][0]
        answer = fs_example['answer']
        fs_lines.append(f"Q: {para} A: {answer}")
    
    # Join few-shot lines with newlines, add trailing newline
    shared_parts.append("\n".join(fs_lines) + "\n")
    
    # Combine shared parts
    shared_part = "\n".join(shared_parts)
    
    # Build individual paraphrase parts
    para_parts = []
    for para in question_paraphrases:
        para_part = f"Q: {para} A:"
        para_parts.append(para_part)
    
    # Return list: [shared_part, para_part_1, para_part_2, ...]
    return [shared_part] + para_parts


def tokenize_prompt_parts(prompt_parts, tokenizer):
    """
    Tokenize prompt parts and track positions for each segment.
    """
    # Tokenize each part individually
    tokenized_parts = []
    separator = "\n"  # Single newline separator
    sep_tokens = tokenizer.encode(separator, add_special_tokens=False)
    
    for part in prompt_parts:
        tokens = tokenizer.encode(part, add_special_tokens=False)
        tokenized_parts.append(tokens)
    
    # Build full token sequence and track positions
    full_tokens = []
    segment_positions = []
    current_pos = 0
    
    for i, tokens in enumerate(tokenized_parts):
        if i > 0:
            # Add separator
            full_tokens.extend(sep_tokens)
            current_pos += len(sep_tokens)
        
        start_pos = current_pos
        full_tokens.extend(tokens)
        current_pos += len(tokens)
        end_pos = current_pos
        
        segment_positions.append((start_pos, end_pos))
    
    # Build question_group_ids tensor
    total_length = len(full_tokens)
    question_group_ids = torch.zeros(total_length, dtype=torch.long)
    
    for group_idx, (start, end) in enumerate(segment_positions):
        question_group_ids[start:end] = group_idx
    
    # Decode back to text
    concatenated_text = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    return concatenated_text, segment_positions, question_group_ids, total_length


class TestBuildQuestionStructMask:
    """Tests for build_question_struct_mask function."""
    
    def test_basic_mask_shape(self):
        """Test that mask has correct shape [1, 1, Q, K]."""
        Q, K = 10, 10
        question_group_ids = torch.zeros(10, dtype=torch.long)
        question_group_ids[5:] = 1  # Last 5 tokens are para_1
        
        mask = build_question_struct_mask(
            question_group_ids, Q, K,
            dtype=torch.float32,
            device=torch.device('cpu')
        )
        
        assert mask.shape == (1, 1, Q, K)
    
    def test_shared_part_sees_all(self):
        """Test that shared part (group 0) can attend to all tokens."""
        Q, K = 10, 10
        question_group_ids = torch.zeros(10, dtype=torch.long)
        question_group_ids[5:8] = 1   # tokens 5-7 are para_1
        question_group_ids[8:] = 2    # tokens 8-9 are para_2
        
        mask = build_question_struct_mask(
            question_group_ids, Q, K,
            dtype=torch.float32,
            device=torch.device('cpu')
        )
        
        # Shared part queries (tokens 0-4) should have 0 mask value for all keys
        for q in range(5):
            for k in range(10):
                assert mask[0, 0, q, k] == 0, f"Shared token {q} should see token {k}"
    
    def test_para_part_sees_shared_and_self(self):
        """Test that para parts can see shared part and themselves."""
        Q, K = 10, 10
        question_group_ids = torch.zeros(10, dtype=torch.long)
        question_group_ids[5:8] = 1   # tokens 5-7 are para_1
        question_group_ids[8:] = 2    # tokens 8-9 are para_2
        
        mask = build_question_struct_mask(
            question_group_ids, Q, K,
            dtype=torch.float32,
            device=torch.device('cpu')
        )
        
        neg_inf = torch.finfo(torch.float32).min
        
        # Para 1 queries (tokens 5-7) should see shared (0-4) and self (5-7)
        for q in range(5, 8):
            # Can see shared part
            for k in range(5):
                assert mask[0, 0, q, k] == 0, f"Para1 token {q} should see shared token {k}"
            # Can see self (para 1)
            for k in range(5, 8):
                assert mask[0, 0, q, k] == 0, f"Para1 token {q} should see para1 token {k}"
            # Cannot see para 2
            for k in range(8, 10):
                assert mask[0, 0, q, k] == neg_inf, f"Para1 token {q} should NOT see para2 token {k}"
        
        # Para 2 queries (tokens 8-9) should see shared (0-4) and self (8-9)
        for q in range(8, 10):
            # Can see shared part
            for k in range(5):
                assert mask[0, 0, q, k] == 0, f"Para2 token {q} should see shared token {k}"
            # Cannot see para 1
            for k in range(5, 8):
                assert mask[0, 0, q, k] == neg_inf, f"Para2 token {q} should NOT see para1 token {k}"
            # Can see self (para 2)
            for k in range(8, 10):
                assert mask[0, 0, q, k] == 0, f"Para2 token {q} should see para2 token {k}"
    
    def test_three_para_parts(self):
        """Test with three para parts to verify isolation."""
        Q, K = 16, 16
        question_group_ids = torch.zeros(16, dtype=torch.long)
        question_group_ids[4:8] = 1    # para_1
        question_group_ids[8:12] = 2   # para_2
        question_group_ids[12:] = 3    # para_3
        
        mask = build_question_struct_mask(
            question_group_ids, Q, K,
            dtype=torch.float32,
            device=torch.device('cpu')
        )
        
        neg_inf = torch.finfo(torch.float32).min
        
        # Verify para_2 (tokens 8-11) isolation
        for q in range(8, 12):
            # Can see shared (0-3)
            for k in range(4):
                assert mask[0, 0, q, k] == 0
            # Cannot see para_1 (4-7)
            for k in range(4, 8):
                assert mask[0, 0, q, k] == neg_inf
            # Can see self (8-11)
            for k in range(8, 12):
                assert mask[0, 0, q, k] == 0
            # Cannot see para_3 (12-15)
            for k in range(12, 16):
                assert mask[0, 0, q, k] == neg_inf


class TestConstructPromptNewFormat:
    """Tests for construct_prompt_new_format function."""
    
    def test_basic_construction(self):
        """Test basic prompt construction."""
        instruction = "Answer the question."
        few_shot_examples = [
            {'paraphrases': ['Q1?', 'Q1 alt?'], 'answer': 'A1'},
            {'paraphrases': ['Q2?'], 'answer': 'A2'},
        ]
        question_paraphrases = ['Main Q1?', 'Main Q2?']
        
        parts = construct_prompt_new_format(instruction, few_shot_examples, question_paraphrases)
        
        # Should have 3 parts: shared + 2 para parts
        assert len(parts) == 3
        
        # Shared part should contain instruction and few-shot
        assert instruction in parts[0]
        assert "Q: Q1? A: A1" in parts[0]
        assert "Q: Q2? A: A2" in parts[0]
        
        # Para parts should be formatted correctly
        assert parts[1] == "Q: Main Q1? A:"
        assert parts[2] == "Q: Main Q2? A:"
    
    def test_empty_paraphrases(self):
        """Test with no paraphrases."""
        instruction = "Answer."
        few_shot_examples = [{'paraphrases': ['Q?'], 'answer': 'A'}]
        question_paraphrases = []
        
        parts = construct_prompt_new_format(instruction, few_shot_examples, question_paraphrases)
        
        # Should have only shared part
        assert len(parts) == 1
        assert instruction in parts[0]
    
    def test_single_paraphrase(self):
        """Test with single paraphrase."""
        instruction = "Answer."
        few_shot_examples = [{'paraphrases': ['Q?'], 'answer': 'A'}]
        question_paraphrases = ['Single question?']
        
        parts = construct_prompt_new_format(instruction, few_shot_examples, question_paraphrases)
        
        assert len(parts) == 2
        assert parts[1] == "Q: Single question? A:"


class TestTokenizePromptParts:
    """Tests for tokenize_prompt_parts function (requires mock tokenizer)."""
    
    def test_with_mock_tokenizer(self):
        """Test tokenization with a simple mock."""
        # Simple mock tokenizer that splits on whitespace
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Return simple token IDs (just indices)
                tokens = text.split()
                return list(range(len(tokens)))
            
            def decode(self, token_ids, skip_special_tokens=False):
                # Just return placeholder text
                return " ".join(["tok"] * len(token_ids))
        
        tokenizer = MockTokenizer()
        prompt_parts = ["shared part", "para one", "para two"]
        
        text, positions, group_ids, length = tokenize_prompt_parts(prompt_parts, tokenizer)
        
        # Check positions are tracked
        assert len(positions) == 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in positions)
        
        # Check group IDs
        assert len(group_ids) == length
        
        # First segment should be group 0
        start0, end0 = positions[0]
        assert all(group_ids[i] == 0 for i in range(start0, end0))
        
        # Second segment should be group 1
        start1, end1 = positions[1]
        assert all(group_ids[i] == 1 for i in range(start1, end1))


class TestMaskDeviceHandling:
    """Tests for device handling in mask building."""
    
    def test_cpu_device(self):
        """Test mask building on CPU."""
        question_group_ids = torch.zeros(10, dtype=torch.long)
        question_group_ids[5:] = 1
        
        mask = build_question_struct_mask(
            question_group_ids, 10, 10,
            dtype=torch.float32,
            device=torch.device('cpu')
        )
        
        assert mask.device == torch.device('cpu')
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test mask building on CUDA."""
        question_group_ids = torch.zeros(10, dtype=torch.long)
        question_group_ids[5:] = 1
        
        mask = build_question_struct_mask(
            question_group_ids, 10, 10,
            dtype=torch.float32,
            device=torch.device('cuda:0')
        )
        
        assert mask.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
