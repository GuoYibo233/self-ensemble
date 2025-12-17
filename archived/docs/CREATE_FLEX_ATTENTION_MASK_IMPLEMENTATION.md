# Implementation Guide: create_flex_attention_mask

## Overview

The `create_flex_attention_mask` function implements segment-based attention masking for multi-paraphrase ensemble generation using PyTorch's FlexAttention API.

## Purpose

This function creates a mask that:
1. **During encoding** (original tokens): Isolates each paraphrase segment - tokens only attend within their own segment
2. **During generation** (new tokens): Allows fusion - generated tokens can attend to all previous tokens
3. **Always**: Respects causal constraint - no attending to future tokens

## Implementation Strategy

### Challenge: vmap Compilation

FlexAttention uses PyTorch's `vmap` (vectorized map) for efficient compilation. This has a critical constraint:

**❌ Data-dependent control flow is NOT allowed:**
```python
# This FAILS - data-dependent loop
def bad_mask(b, h, q_idx, kv_idx):
    for seg in segments:  # ❌ Loop over data
        if seg['start'] <= q_idx < seg['end']:  # ❌ Data-dependent if
            ...
    return result  # ❌ May return Python bool
```

**✅ Tensor operations ARE allowed:**
```python
# This WORKS - pure tensor operations
def good_mask(b, h, q_idx, kv_idx):
    causal = q_idx >= kv_idx  # ✅ Tensor comparison
    in_segment = (q_idx >= starts) & (q_idx < ends)  # ✅ Tensor ops
    return causal & in_segment.any()  # ✅ Returns Tensor[bool]
```

### Solution: Tensor-Based Implementation

```python
def create_flex_attention_mask(segment_positions, original_length):
    import torch
    
    # 1. Convert segment boundaries to tensors (done once)
    segment_starts = torch.tensor([start for start, _ in segment_positions], dtype=torch.int64)
    segment_ends = torch.tensor([end for _, end in segment_positions], dtype=torch.int64)
    
    def mask_mod(b, h, q_idx, kv_idx):
        # 2. All operations use tensor comparisons
        causal_mask = q_idx >= kv_idx  # Tensor[bool]
        is_generated = q_idx >= original_length  # Tensor[bool]
        
        # 3. Check segment membership using broadcasting
        q_in_segment = (q_idx >= segment_starts) & (q_idx < segment_ends)  # Tensor[num_segments]
        kv_in_segment = (kv_idx >= segment_starts) & (kv_idx < segment_ends)  # Tensor[num_segments]
        
        # 4. Both in same segment if any position matches
        same_segment = (q_in_segment & kv_in_segment).any()  # Tensor[bool]
        
        # 5. Combine constraints
        result = causal_mask & (is_generated | same_segment)  # Tensor[bool]
        
        return result  # ✅ Always returns Tensor[bool]
    
    return mask_mod
```

## Data Types and Shapes

### Input Types

```python
segment_positions: List[(int, int)]
    Example: [(0, 48), (48, 95), (95, 143), (143, 192), (192, 238)]
    
original_length: int
    Example: 238

# Inside mask_mod:
b: Tensor (scalar)      # Batch index
h: Tensor (scalar)      # Head index  
q_idx: Tensor (scalar)  # Query position
kv_idx: Tensor (scalar) # Key position
```

### Internal Tensors

```python
segment_starts: Tensor[int64]
    Shape: [num_segments]
    Example: tensor([  0,  48,  95, 143, 192])

segment_ends: Tensor[int64]
    Shape: [num_segments]
    Example: tensor([ 48,  95, 143, 192, 238])

# Inside mask_mod:
causal_mask: Tensor[bool]
    Shape: []  (scalar)
    Example: tensor(True)

is_generated: Tensor[bool]
    Shape: []  (scalar)
    Example: tensor(False)

q_in_segment: Tensor[bool]
    Shape: [num_segments]
    Example: tensor([True, False, False, False, False])

kv_in_segment: Tensor[bool]
    Shape: [num_segments]
    Example: tensor([True, False, False, False, False])

same_segment: Tensor[bool]
    Shape: []  (scalar)
    Example: tensor(True)

result: Tensor[bool]
    Shape: []  (scalar)
    Example: tensor(True)
```

## Masking Logic

### Encoding Phase (q_idx < original_length)

For a token at position `q` in segment `i`:
- ✅ Can attend to tokens in segment `i` (same segment)
- ❌ Cannot attend to tokens in other segments
- ❌ Cannot attend to future tokens (causal)

```
Segment 1: [0-47]     → Attends to [0-47]
Segment 2: [48-94]    → Attends to [48-94]
Segment 3: [95-142]   → Attends to [95-142]
...
```

### Generation Phase (q_idx >= original_length)

For a generated token at position `q`:
- ✅ Can attend to ALL segments
- ✅ Can attend to previous generated tokens
- ❌ Cannot attend to future tokens (causal)

```
Generated[238]: → Attends to [0-238]
Generated[239]: → Attends to [0-239]
Generated[240]: → Attends to [0-240]
...
```

### Visual Example

```
Sequence: [S1....][S2....][S3....][G1][G2][G3]
          0    47 48   94 95  142 238 239 240

Position 50 (in S2):
  ✅ Can attend to: [48, 49, 50] (same segment, causal)
  ❌ Cannot attend to: [0-47] (different segment)
  ❌ Cannot attend to: [51-94] (future in same segment)
  ❌ Cannot attend to: [95+] (different segment or future)

Position 238 (first generated):
  ✅ Can attend to: [0-238] (all segments, causal)
  ❌ Cannot attend to: [239+] (future)
```

## Testing Strategy

### Unit Tests (without PyTorch installed)

The implementation can be validated through code inspection:

1. **Type Safety**: All operations return `Tensor[bool]`
   - `>=`, `<`, `&`, `|` on tensors return tensors
   - `.any()` on tensor returns scalar tensor

2. **No Data-Dependent Control**: 
   - No Python `for` loops over tensor values
   - No `if` statements on tensor values
   - Only tensor operations

3. **Correctness**:
   - `causal_mask`: Ensures `q >= kv`
   - `is_generated`: Checks if `q >= original_length`
   - `same_segment`: Verifies both q and kv in same segment
   - `result`: Combines all constraints with logical AND/OR

### Integration Tests (with PyTorch)

Run `test_create_flex_attention_mask.py`:
```bash
python3 test_create_flex_attention_mask.py
```

This validates:
- ✅ Returns `Tensor[bool]`, not Python `bool`
- ✅ Causal constraint enforced
- ✅ Segment isolation during encoding
- ✅ Full attention during generation
- ✅ Edge cases handled correctly

### Visual Verification

Run `test_mask_visualization.py` to see the mask pattern:
```bash
python3 test_mask_visualization.py
```

Expected output shows:
- Diagonal blocks for segments (encoding isolation)
- Full rows for generated tokens (generation fusion)

## Common Issues and Solutions

### Issue 1: Python bool returned

**Problem**: `return True` or `return q_idx > kv_idx` may return Python bool

**Solution**: Ensure all comparisons use tensors
```python
# ❌ Bad
return True

# ✅ Good  
return q_idx >= kv_idx  # Returns Tensor[bool]
```

### Issue 2: Data-dependent control flow

**Problem**: Loops or conditionals on tensor values

**Solution**: Use tensor operations
```python
# ❌ Bad
for i, (start, end) in enumerate(segment_positions):
    if start <= q_idx < end:
        ...

# ✅ Good
q_in_segment = (q_idx >= segment_starts) & (q_idx < segment_ends)
```

### Issue 3: Shape mismatches

**Problem**: Broadcasting issues with different tensor shapes

**Solution**: Ensure compatible shapes
```python
# segment_starts: [num_segments]
# q_idx: scalar
# Result: [num_segments] (broadcasts correctly)
q_in_segment = q_idx >= segment_starts
```

## Performance Considerations

1. **One-time Conversion**: `segment_starts` and `segment_ends` created once, reused for all mask calls
2. **Tensor Operations**: Highly optimized by PyTorch, GPU-friendly
3. **No Python Loops**: All work done in compiled tensor operations
4. **vmap Compatible**: Can be vectorized efficiently

## References

- **PyTorch FlexAttention Blog**: https://pytorch.org/blog/flexattention/
- **attention-gym Repository**: https://github.com/meta-pytorch/attention-gym
- **PyTorch vmap Documentation**: https://pytorch.org/docs/stable/generated/torch.vmap.html

## Integration with FlexAttentionWrapper

The mask is used in `create_patched_forward`:

```python
# Create mask
mask_mod = create_flex_attention_mask(segment_positions, original_length)

# Create block mask
block_mask = create_block_mask(
    mask_mod,  # Our segment-based mask function
    B=bsz, H=num_heads, 
    Q_LEN=q_len, KV_LEN=q_len,
    device=device
)

# Apply FlexAttention
attn_output = flex_attention(
    query_states, key_states, value_states,
    block_mask=block_mask
)
```

## Validation Checklist

When implementing or modifying this function, verify:

- [ ] Returns `Tensor[bool]`, never Python `bool`
- [ ] No data-dependent loops or conditionals
- [ ] Only uses tensor operations (>=, <, &, |, .any())
- [ ] Causal constraint always enforced
- [ ] Segment isolation during encoding
- [ ] Full attention during generation
- [ ] Works with scalar tensor indices
- [ ] No shape mismatches
- [ ] Documented with clear comments
- [ ] Tested with unit and integration tests

## Summary

The implementation successfully:
✅ Uses tensor operations to avoid vmap issues
✅ Returns proper `Tensor[bool]` types
✅ Implements segment isolation for encoding
✅ Enables fusion during generation
✅ Maintains causal constraint
✅ Is compatible with FlexAttention API
✅ Is efficient and GPU-friendly

This provides the foundation for multi-paraphrase ensemble generation with attention-based fusion.
