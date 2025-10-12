# FlexAttention Architecture Diagram

## 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT PREPARATION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Question: "What is the capital of France?"                      │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Generate 5 Paraphrases (using dataset.construct_prompts) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  Para1: "Q: What is France's capital city? A:"                  │
│  Para2: "Q: Tell me the capital of France. A:"                  │
│  Para3: "Q: Which city is France's capital? A:"                 │
│  Para4: "Q: What city serves as France's capital? A:"           │
│  Para5: "Q: Can you name France's capital? A:"                  │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   concatenate_paraphrases_with_positions()               │   │
│  │   (NEW FUNCTION)                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  Concatenated:                                                   │
│  "Q: What is France's capital city? A: [SEP] Q: Tell me..."     │
│                                                                   │
│  Positions: [(0,45), (50,92), (97,140), (145,195), (200,245)]  │
│  Original Length: 245 tokens                                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ENCODING PHASE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Token Positions:  0    ...   45  50  ...  92  97  ... 245      │
│  Segments:        [─ Para1 ─][─ Para2 ─][─ Para3 ─] ...        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  FlexAttention Mask (create_segment_isolation_mask)     │    │
│  │  (NEW FUNCTION)                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  Attention Matrix (Simplified):                                  │
│                                                                   │
│          kv→ Para1  Para2  Para3  Para4  Para5                  │
│       q↓  ┌──────┬──────┬──────┬──────┬──────┐                 │
│    Para1  │  ✓✓  │  ✗✗  │  ✗✗  │  ✗✗  │  ✗✗  │  (Isolated)   │
│           ├──────┼──────┼──────┼──────┼──────┤                 │
│    Para2  │  ✗✗  │  ✓✓  │  ✗✗  │  ✗✗  │  ✗✗  │  (Isolated)   │
│           ├──────┼──────┼──────┼──────┼──────┤                 │
│    Para3  │  ✗✗  │  ✗✗  │  ✓✓  │  ✗✗  │  ✗✗  │  (Isolated)   │
│           ├──────┼──────┼──────┼──────┼──────┤                 │
│    Para4  │  ✗✗  │  ✗✗  │  ✗✗  │  ✓✓  │  ✗✗  │  (Isolated)   │
│           ├──────┼──────┼──────┼──────┼──────┤                 │
│    Para5  │  ✗✗  │  ✗✗  │  ✗✗  │  ✗✗  │  ✓✓  │  (Isolated)   │
│           └──────┴──────┴──────┴──────┴──────┘                 │
│                                                                   │
│  ✓✓ = Can attend (within same segment, causal)                  │
│  ✗✗ = Cannot attend (different segments)                        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   GENERATION PHASE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Generate Token 1 (position 246):                                │
│                                                                   │
│          kv→ Para1  Para2  Para3  Para4  Para5  Gen              │
│       q↓  ┌──────┬──────┬──────┬──────┬──────┬─────┐           │
│    Gen1   │  ✓✓  │  ✓✓  │  ✓✓  │  ✓✓  │  ✓✓  │ (1) │ (Fusion) │
│           └──────┴──────┴──────┴──────┴──────┴─────┘           │
│                                                                   │
│  Generate Token 2 (position 247):                                │
│                                                                   │
│          kv→ Para1  Para2  Para3  Para4  Para5  Gen1  Gen2      │
│       q↓  ┌──────┬──────┬──────┬──────┬──────┬─────┬─────┐     │
│    Gen2   │  ✓✓  │  ✓✓  │  ✓✓  │  ✓✓  │  ✓✓  │ ✓✓  │ (2) │     │
│           └──────┴──────┴──────┴──────┴──────┴─────┴─────┘     │
│                                                                   │
│  ✓✓ = Can attend to ALL previous tokens (fusion)                │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  flex_attention_generation() Loop:                       │    │
│  │  (NEW FUNCTION, reuses generation loop structure)        │    │
│  │                                                           │    │
│  │  for step in range(max_new_tokens):                      │    │
│  │      1. Create mask for current length                   │    │
│  │      2. Patch model with FlexAttention                   │    │
│  │      3. Forward pass → get logits                        │    │
│  │      4. Unpatch model                                    │    │
│  │      5. Select next token (argmax)  ← REUSED             │    │
│  │      6. Append to sequence          ← REUSED             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Output: "Paris"                                                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Code Reuse Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                  flex_attention_generate.py                      │
│                                                                   │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃ REUSED FROM generate.py (54% of code)                     ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Lemmatization Functions (Lines 48-77)                    │    │
│  │  • init_spacy()                                          │    │
│  │  • lemmaize_predicts()                                   │    │
│  │  • lemmaize_chunk()                                      │    │
│  │  • append_lemmas()                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Main Script Structure (Lines 380-554)                    │    │
│  │  • Argument parser                                       │    │
│  │  • Dataset loading                                       │    │
│  │  • Model loading                                         │    │
│  │  • Main loop structure                                   │    │
│  │  • Result storage                                        │    │
│  │  • File saving                                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Generation Patterns in flex_attention_generation()       │    │
│  │  • Model config setup                                    │    │
│  │  • Tokenization                                          │    │
│  │  • Generation loop structure                             │    │
│  │  • Token selection (argmax)                              │    │
│  │  • Sequence update                                       │    │
│  │  • Output decoding                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃ NEW IMPLEMENTATION (46% of code)                          ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ concatenate_paraphrases_with_positions() (Lines 83-130)  │    │
│  │  • Tokenize each prompt separately                       │    │
│  │  • Build full token sequence with separators             │    │
│  │  • Track segment positions                               │    │
│  │  • Return (text, positions, length)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ create_segment_isolation_mask() (Lines 132-185)          │    │
│  │  • Create mask_mod function                              │    │
│  │  • Implement 3 mask rules:                               │    │
│  │    1. Causal constraint                                  │    │
│  │    2. Segment isolation (encoding)                       │    │
│  │    3. Full attention (generation)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ FlexAttentionWrapper Class (Lines 187-306)               │    │
│  │  • __init__(): Initialize wrapper                        │    │
│  │  • patch_model(): Replace attention with FlexAttention   │    │
│  │  • unpatch_model(): Restore original attention           │    │
│  │  • create_patched_forward(): Custom forward function     │    │
│  │    - Compute Q, K, V                                     │    │
│  │    - Apply RoPE if needed                                │    │
│  │    - Create block_mask                                   │    │
│  │    - Call flex_attention()                               │    │
│  │    - Fallback to SDPA on error                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ flex_attention_generation() (Lines 308-378)              │    │
│  │  • Orchestrate the entire generation process             │    │
│  │  • Call concatenate_paraphrases_with_positions()         │    │
│  │  • Create FlexAttentionWrapper                           │    │
│  │  • Generation loop with dynamic mask updates             │    │
│  │  • Patch/unpatch at each step                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Comparison with generate.py Methods

```
┌──────────────────────────────────────────────────────────────────┐
│                    GENERATION METHOD COMPARISON                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. per_prompt (generate.py)                                      │
│  ─────────────────────────────                                   │
│                                                                    │
│  Para1 ──→ Model ──→ Output1 ──┐                                 │
│  Para2 ──→ Model ──→ Output2   │ Store all outputs               │
│  Para3 ──→ Model ──→ Output3   │ separately                      │
│  Para4 ──→ Model ──→ Output4   │                                 │
│  Para5 ──→ Model ──→ Output5 ──┘                                 │
│                                                                    │
│  • 5 separate forward passes                                      │
│  • No fusion                                                       │
│  • Baseline for comparison                                        │
│                                                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  2. avg / max / weighted_avg (generate.py)                        │
│  ──────────────────────────────────────────                      │
│                                                                    │
│  Para1 ──→ Model ──→ Logits1 ──┐                                 │
│  Para2 ──→ Model ──→ Logits2   │                                 │
│  Para3 ──→ Model ──→ Logits3   ├─→ Fusion ──→ Token              │
│  Para4 ──→ Model ──→ Logits4   │   (avg/max)                     │
│  Para5 ──→ Model ──→ Logits5 ──┘                                 │
│                                                                    │
│  • 5 separate forward passes per step                             │
│  • Fusion at logit level                                          │
│  • All paras see same previous tokens                             │
│                                                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  3. flex_attention (NEW)                                          │
│  ────────────────────────                                        │
│                                                                    │
│  [Para1 SEP Para2 SEP Para3 SEP Para4 SEP Para5]                 │
│                        ↓                                          │
│              FlexAttention Model                                  │
│              (with segment isolation)                             │
│                        ↓                                          │
│                    Single Token                                   │
│                                                                    │
│  • 1 forward pass per step                                        │
│  • Fusion at attention level                                      │
│  • Paras are isolated during encoding                             │
│  • Generated tokens see all paras                                 │
│                                                                    │
│  Advantages:                                                      │
│  ✓ More efficient (1 forward pass vs 5)                          │
│  ✓ More flexible fusion (attention-based)                         │
│  ✓ Better context integration                                     │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

## 4. FlexAttention Mask Visualization

```
Sequence: [Para1 Para1 Para1] [SEP] [Para2 Para2 Para2] [SEP] ... [Gen1 Gen2 Gen3]
Position:  0     1     2      3      4     5     6      7          15   16   17

Attention Matrix (✓ = can attend, ✗ = cannot attend):

         KV→  0   1   2   3   4   5   6   7  ...  15  16  17
       Q↓   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
       0    │ ✓ │   │   │   │   │   │   │   │   │   │   │   │  Para1
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       1    │ ✓ │ ✓ │   │   │   │   │   │   │   │   │   │   │  Para1
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       2    │ ✓ │ ✓ │ ✓ │   │   │   │   │   │   │   │   │   │  Para1
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       3    │ ✗ │ ✗ │ ✗ │ ✓ │   │   │   │   │   │   │   │   │  [SEP]
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       4    │ ✗ │ ✗ │ ✗ │ ✗ │ ✓ │   │   │   │   │   │   │   │  Para2
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       5    │ ✗ │ ✗ │ ✗ │ ✗ │ ✓ │ ✓ │   │   │   │   │   │   │  Para2
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       6    │ ✗ │ ✗ │ ✗ │ ✗ │ ✓ │ ✓ │ ✓ │   │   │   │   │   │  Para2
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       7    │ ✗ │ ✗ │ ✗ │ ✗ │ ✗ │ ✗ │ ✗ │ ✓ │   │   │   │   │  [SEP]
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
      ...   │   │   │   │   │   │   │   │   │...│   │   │   │
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       15   │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │...│ ✓ │   │   │  Gen1
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       16   │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │...│ ✓ │ ✓ │   │  Gen2
            ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       17   │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │...│ ✓ │ ✓ │ ✓ │  Gen3
            └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

Key Observations:
• Upper triangle is always ✗ (causal constraint)
• Original segments form isolated diagonal blocks
• Generated tokens (row 15+) can attend to all previous tokens
```

## 5. File Organization

```
self-ensemble/
├── generate.py                         ← Original implementation
│   ├── per_prompt mode
│   ├── avg/max/weighted modes
│   └── Lemmatization utilities
│
├── flex_attention_generate.py          ← NEW implementation
│   ├── Reused functions (54%)
│   │   ├── init_spacy()
│   │   ├── lemmaize_predicts()
│   │   ├── lemmaize_chunk()
│   │   └── append_lemmas()
│   │
│   └── New functions (46%)
│       ├── concatenate_paraphrases_with_positions()
│       ├── create_segment_isolation_mask()
│       ├── FlexAttentionWrapper class
│       └── flex_attention_generation()
│
├── FLEX_ATTENTION_IMPLEMENTATION.md    ← English documentation
│   ├── What's reused vs new
│   ├── Technical details
│   └── Usage examples
│
└── 实现总结.md                         ← Chinese summary
    ├── 复用内容详解
    ├── 新实现内容详解
    └── 架构对比
```
