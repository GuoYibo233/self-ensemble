# Custom per-query attention for LLaMA & Qwen (no FlexAttention)

Goal:  
- I have **one concatenated prompt** with multiple question paraphrases.  
- I want an attention pattern where:

- **Q1 tokens** → can only see **Q1 tokens** (plus padding/causal rules as usual)  
- **Q2–Q6 tokens** → can see **their own question tokens** AND **Q1 tokens**  
- Everything else (instruction, few-shot, etc.) keeps **normal causal attention**.

This should replace the old `FlexAttentionWrapper` / `mask_mod` logic and work on the standard HF attention path.

---

## 1. Represent structure at token level

After building the full prompt (instruction + few-shot + all paraphrases) and tokenizing it once:

```python
# full concatenated input for one sample
inputs = tokenizer(concatenated_text, return_tensors="pt", add_special_tokens=True)
input_ids = inputs["input_ids"].to(model.device)      # [1, S]
S = input_ids.size(1)

# For each token position t (0..S-1), assign a question group:
#   0  → not in any question (instruction / few-shot / separators)
#   1  → main question paraphrase group 1
#   2  → main question paraphrase group 2
#   ...
#   6  → main question paraphrase group 6
question_group_ids = torch.zeros(S, dtype=torch.long, device=model.device)  # [S]

# Fill this based on your segment parsing:
# e.g. using segment_positions/segment_metadata from your code:
for (start, end), meta in zip(segment_positions, segment_metadata):
    if meta["type"] == "question":
        g = 1 + meta["paraphrase_idx"]  # paraphrase_idx: 0..5 → group 1..6
        question_group_ids[start:end] = g

# Store on model so the mask hook can see it
model.question_group_ids = question_group_ids  # [S]
```

---

## 2. Build the per-query **structure mask** `[1, 1, Q, K]`

Rule (token-level):

- Let `g_q = question_group_ids[q]`, `g_k = question_group_ids[k]`.
- If `g_q == 1` (Q1 token): allow only `g_k == 1`.
- If `g_q in {2..6}`: allow `g_k == g_q` or `g_k == 1`.
- If `g_q == 0` (instruction/few-shot/etc.): allow everything (structure doesn’t constrain it).

```python
def build_question_struct_mask(question_group_ids, Q, K, dtype, device):
    """question_group_ids: [S] with values in {0..6}, aligned to sequence positions.
    We assume Q == K == S for the prompt encoding step.
    Returns: [1, 1, Q, K] additive mask (0 or -inf)."""
    import torch

    neg_inf = torch.finfo(dtype).min

    # [Q,1] and [1,K]
    g_q = question_group_ids[:Q].view(Q, 1)  # [Q,1]
    g_k = question_group_ids[:K].view(1, K)  # [1,K]

    # boolean conditions
    is_q1      = (g_q == 1)
    is_q_other = (g_q >= 2) & (g_q <= 6)
    is_ctx     = (g_q == 0)

    # For q in Q1: only Q1 keys
    allowed_q1 = is_q1 & (g_k == 1)

    # For q in Q2..Q6: keys in same group OR group 1
    same_group = (g_q == g_k)
    is_k_q1    = (g_k == 1)
    allowed_qk = is_q_other & (same_group | is_k_q1)

    # For q in context (0): no structural restriction → allow all
    allowed_ctx = is_ctx.repeat(1, K)  # broadcast across keys

    allowed = allowed_q1 | allowed_qk | allowed_ctx  # [Q,K] boolean

    mask = torch.zeros((1, 1, Q, K), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, neg_inf)
    return mask
```

You will **add** this structure mask on top of the model’s existing causal/padding mask.

---

## 3. LLaMA: override `_update_causal_mask` (no FlexAttention)

For LLaMA (HF `modeling_llama`):

- Force eager attention so the explicit mask is always used:

```python
model.config._attn_implementation = "eager"
```

- Wrap `LlamaModel._update_causal_mask`:

```python
from transformers.models.llama.modeling_llama import LlamaModel

def install_llama_struct_mask(model):
    base: LlamaModel = model.model
    old_update = base._update_causal_mask

    def _update_causal_mask_patch(attention_mask, input_tensor, cache_position,
                                  past_key_values, output_attentions):
        # base 4D mask: [B, 1, Q, K]
        causal_mask = old_update(attention_mask, input_tensor,
                                 cache_position, past_key_values,
                                 output_attentions)
        if causal_mask is None:
            return None  # SDPA path; with eager this should not happen

        B, H1, Q, K = causal_mask.shape
        assert B == 1, "MyriadLAMA pipeline uses batch_size=1"

        struct_mask = build_question_struct_mask(
            model.question_group_ids, Q, K,
            dtype=causal_mask.dtype,
            device=causal_mask.device,
        )  # [1,1,Q,K]

        if H1 > 1:
            struct_mask = struct_mask.expand(B, H1, Q, K)

        return causal_mask + struct_mask

    base._update_causal_mask = _update_causal_mask_patch
```

Call this once after loading the model, **before** generation:

```python
install_llama_struct_mask(model)
```

Then, for each prompt, set `model.question_group_ids` from your segment parsing before calling `model(...)` or `generate(...)`.

---

## 4. Qwen2 / Qwen3: patch decoder layer `forward`

For Qwen, we patch each decoder layer’s `forward` so we can tweak the 4D `attention_mask` right before attention is called.

Again, force eager attention:

```python
model.config._attn_implementation = "eager"
```

Then:

```python
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer

def install_qwen_struct_mask(model):
    base: Qwen2Model = model.model

    for layer_idx, layer in enumerate(base.layers):
        assert isinstance(layer, Qwen2DecoderLayer)
        old_forward = layer.forward

        def make_new_forward(old_fwd):
            def new_forward(hidden_states,
                            attention_mask=None,
                            position_ids=None,
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=False,
                            cache_position=None,
                            position_embeddings=None,
                            **kwargs):
                if attention_mask is not None and attention_mask.dim() == 4:
                    B, H1, Q, K = attention_mask.shape
                    assert B == 1

                    struct_mask = build_question_struct_mask(
                        model.question_group_ids, Q, K,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )  # [1,1,Q,K]

                    if H1 > 1:
                        struct_mask = struct_mask.expand(B, H1, Q, K)

                    attention_mask = attention_mask + struct_mask

                return old_fwd(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            return new_forward

        layer.forward = make_new_forward(old_forward)
```

Install once after loading:

```python
install_qwen_struct_mask(model)
```

Again, before each forward/generation you must set:

```python
model.question_group_ids = question_group_ids  # [S]
```

---

## 5. Integration notes (for Copilot to infer)

- This replaces the old FlexAttention path (`FlexAttentionWrapper`, `mask_mod`, `create_block_mask`) and uses the **standard HF attention** (`scaled_dot_product_attention`) with a custom 4D `attention_mask`.
- `question_group_ids` should be built from the existing `segment_positions` / `segment_metadata` pipeline you already have:
  - mark non-question segments as `0`
  - mark main question paraphrases by `1..N`
- The mask is **additive**:  
  - allowed positions = `0`  
  - blocked positions = large negative (e.g. `torch.finfo(dtype).min`)
- This preserves the standard causal constraint (no “look into the future”), because we always **add** our structure mask on top of the base causal mask.

With this, Copilot has enough context to generate the necessary glue code and adapt it to both LLaMA and Qwen with your “Q1-only / Q2–Q6 see Q1+self” structure.
