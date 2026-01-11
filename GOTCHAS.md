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
