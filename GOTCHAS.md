do# Gotchas - Qwen3 MoE Implementation

A running list of bugs and mistakes made while implementing Qwen3 MoE from scratch.

## RoPE (Rotary Position Embeddings)

### 1. Wrong dimension in `apply_rope` concatenation
**Bug:** Used `dim=2` instead of `dim=-1` for the rotated tensor concatenation.
```python
# Wrong
rotated_x = torch.cat([-x[..., head_dim // 2:], x[..., :head_dim // 2]], dim=2)

# Correct
rotated_x = torch.cat([-x[..., head_dim // 2:], x[..., :head_dim // 2]], dim=-1)
```
**Why:** The head_dim is always the last dimension, regardless of tensor shape. Using `dim=-1` works for any input shape.

## Attention

### 2. RMSNorm size for q_norm/k_norm
**Bug:** Initialized `q_norm` and `k_norm` with `hidden_size` instead of `head_dim`.
```python
# Wrong
self.q_norm = nn.RMSNorm(self.hidden_size)

# Correct
self.q_norm = nn.RMSNorm(self.head_dim)
```
**Why:** Qwen3 applies QK normalization per-head, so the norm is over `head_dim`, not the full `hidden_size`.

### 3. Missing `dim=-1` in softmax
**Bug:** Called `F.softmax(...)` without specifying the dimension.
```python
# Wrong (implicit dimension warning)
F.softmax(scores)

# Correct
F.softmax(scores, dim=-1)
```
**Why:** PyTorch's softmax requires explicit dimension. For attention, we softmax over the key dimension (last dim).

### 4. Missing transpose before reshape in attention output
**Bug:** Reshaped attention output directly without transposing first.
```python
# Wrong - o has shape (bsz, num_heads, seq, head_dim)
return self.o_proj(o.reshape(bsz, seq, self.hidden_size))

# Correct
return self.o_proj(o.transpose(1, 2).reshape(bsz, seq, self.hidden_size))
```
**Why:** After attention, the tensor is `(bsz, num_heads, seq, head_dim)`. Must transpose to `(bsz, seq, num_heads, head_dim)` before reshaping to `(bsz, seq, hidden_size)`.

### 5. Linear layers with bias when Qwen3 uses no bias
**Bug:** Used default `nn.Linear()` which has `bias=True`.
```python
# Wrong
self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)

# Correct
self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
```
**Why:** Qwen3 uses `attention_bias=False` in its config. All q/k/v/o projections should have no bias.

## Testing

### 6. HuggingFace attention doesn't apply causal mask when `attention_mask=None`
**Bug:** Assumed HF would apply causal masking automatically.
```python
# This does NOT apply causal masking in HF!
hf_out, _ = hf_attn(x, position_embeddings=(cos, sin), attention_mask=None)
```
**Why:** HF's `eager_attention_forward` only applies masking when `attention_mask` is explicitly provided. For testing against an implementation with built-in causal mask, you must pass a causal mask to HF:
```python
def make_causal_mask(seq_len, batch_size):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.float().masked_fill(mask, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
```

## MoE (Mixture of Experts)

### 7. Slice syntax: comma vs colon indexing

**Date**: 2026-01-13
**Component**: Experts
**Severity**: High

**Bug:** Used comma instead of colon for tensor slicing.
```python
# Wrong - this is multi-indexing, returns a scalar
chunk = expert_tokens[chunk_start, chunk_end]

# Correct - this is slicing, returns a 2D tensor
chunk = expert_tokens[chunk_start:chunk_end]
```
**Why:** `tensor[a, b]` indexes two dimensions (returns 0D scalar or broadcasts). `tensor[a:b]` slices along dim 0 (returns chunk of rows).

**Test that catches it:** `RuntimeError: both arguments to matmul need to be at least 1D, but they are 0D and 2D`

---

### 8. SwiGLU: gate and up projections swapped

**Date**: 2026-01-13
**Component**: Experts
**Severity**: High

**Bug:** Applied silu to up_proj instead of gate_proj.
```python
# Wrong - silu on up, multiply by gate
F.silu(chunk @ self.up_proj[i]) * (chunk @ self.gate_proj[i])

# Correct - silu on gate, multiply by up
F.silu(chunk @ self.gate_proj[i]) * (chunk @ self.up_proj[i])
```
**Why:** SwiGLU = `silu(gate(x)) * up(x)`. The gate controls flow, up provides values. Swapping them gives wrong numerical results.

**Test that catches it:** `test_single_expert_vs_hf` - numerical mismatch against HF reference.

---

### 9. Concat dimension for token batches

**Date**: 2026-01-13
**Component**: Experts
**Severity**: High

**Bug:** Concatenated expert outputs along feature dimension instead of token dimension.
```python
# Wrong - concatenates along hidden_size (creates huge tensor)
torch.concat(output, dim=-1)

# Correct - concatenates along token dimension
torch.concat(output, dim=0)
```
**Why:** Each expert processes a chunk of tokens with shape `(num_tokens, hidden_size)`. We want to stack tokens vertically, not widen the hidden dimension.

**Test that catches it:** `RuntimeError: Sizes of tensors must match except in dimension 1`

---

### 10. Expert token counts: wrong sum dimension

**Date**: 2026-01-13
**Component**: MOE
**Severity**: High

**Bug:** Summed routing map along wrong dimension.
```python
# flat_routing_map shape: (num_experts, batch*seq)

# Wrong - sums per position, shape (batch*seq,)
expert_tokens_count = flat_routing_map.sum(dim=0)

# Correct - sums per expert, shape (num_experts,)
expert_tokens_count = flat_routing_map.sum(dim=1)
```
**Why:** We need count of tokens per expert for chunking. `sum(dim=1)` collapses the token dimension, giving per-expert counts.

**Additional issue:** Counts need to be **cumulative** if using as chunk boundaries.

---

### 11. Router normalization: all probs vs top-k only

**Date**: 2026-01-13
**Component**: Router
**Severity**: Medium

**Bug:** Normalized all routing probabilities instead of just the selected top-k.
```python
# Wrong - normalizes all probs (redundant after softmax)
routing_probs = routing_probs / routing_probs.sum(-1, keepdim=True)

# Correct - only normalize the top-k weights
# (done during expert output combination, not in router)
```
**Why:** HuggingFace normalizes only the top-k weights so they sum to 1. Normalizing all probs is redundant (softmax already sums to 1) and doesn't match HF behavior.

**Test that catches it:** `test_router_weights_vs_hf` - top-k weight mismatch.

---

### 12. Missing dim=-1 in Router softmax

**Date**: 2026-01-13
**Component**: Router
**Severity**: High

**Bug:** Called softmax without explicit dimension, causing different expert selection.
```python
# Wrong - implicit dimension choice (deprecated warning)
routing_probs = F.softmax(self.gate(x))

# Correct - explicit last dimension
routing_probs = F.softmax(self.gate(x), dim=-1)
```
**Why:** Without `dim=-1`, PyTorch uses the last dimension by default but behavior can be inconsistent with 3D inputs. This caused the router to select different experts than HF, even with identical weights.

**Test that catches it:** `test_router_weights_vs_hf` - expert selection mismatch.

---

### 13. Expert token counts need cumsum for chunk boundaries

**Date**: 2026-01-13
**Component**: MOE
**Severity**: High

**Bug:** Used per-expert counts directly as chunk boundaries instead of cumulative counts.
```python
# Wrong - gives per-expert counts [128, 130, 127, 127]
expert_tokens_count = flat_routing_map.sum(dim=1)

# Correct - cumulative for chunk boundaries [128, 258, 385, 512]
expert_tokens_count = flat_routing_map.sum(dim=1).cumsum(0)
```
**Why:** When chunking tokens for each expert, we need cumulative indices. With counts `[128, 130, ...]`, expert 1 should start at index 128 (not 130). Without cumsum, chunk boundaries overlap incorrectly.

**Test that catches it:** `test_moe_output_vs_hf` - numerical mismatch due to wrong token-expert assignment.

---

### 14. scatter_add index must match source dimensions

**Date**: 2026-01-13
**Component**: MOE
**Severity**: High

**Bug:** Index tensor had wrong shape for scatter_add operation.
```python
# Wrong - index is (N, 1), source is (N, hidden_size)
output = torch.zeros(...).scatter_add_(0, expert_token_idxs.unsqueeze(-1), ffn_tokens)

# Correct - expand index to match source shape
output = torch.zeros(...).scatter_add_(
    0,
    expert_token_idxs.unsqueeze(-1).expand(-1, hidden_size),
    ffn_tokens
)
```
**Why:** `scatter_add_` requires the index tensor to have the same number of dimensions as the source tensor. The index must be expanded to `(N, hidden_size)` to match `ffn_tokens`.

**Test that catches it:**
- Forward: `RuntimeError: Index tensor must have the same number of dimensions as self tensor`
- Backward: `RuntimeError: ScatterAddBackward0 returned invalid gradient shape [512, 1] expected [512, 256]`

---

### 15. Missing device placement for output tensor

**Date**: 2026-01-13
**Component**: MOE
**Severity**: Medium

**Bug:** Created output tensor on CPU while other tensors were on GPU.
```python
# Wrong - tensor created on CPU
output = torch.zeros(bsz * seq, hidden_size).scatter_add_(...)

# Correct - match input device
output = torch.zeros(bsz * seq, hidden_size, device=x.device).scatter_add_(...)
```
**Why:** When running on GPU, all tensors in an operation must be on the same device. Creating zeros without specifying device defaults to CPU.

**Test that catches it:** `RuntimeError: Expected all tensors to be on the same device`

---

### 16. norm_topk_prob config flag controls weight normalization

**Date**: 2026-01-13
**Component**: Router
**Severity**: High

**Bug:** Always normalized top-k weights regardless of config setting.
```python
# Wrong - always normalizes
routing_probs = routing_probs / top_k_probs.sum(-1, keepdim=True)

# Correct - respect config flag
if self.config.norm_topk_prob:
    routing_probs = routing_probs / top_k_probs.sum(-1, keepdim=True)
```
**Why:** HuggingFace Qwen3 MoE has a `norm_topk_prob` config flag (default: False in some configs). If your implementation always normalizes but HF doesn't (or vice versa), outputs will differ by a consistent scaling factor per token.

**Test that catches it:** `test_moe_output_vs_hf` - outputs differ by ratio of ~1.66x (1/sum_of_topk_weights).

**Debugging tip:** If you see consistent per-token scaling differences, check the `norm_topk_prob` config value.
