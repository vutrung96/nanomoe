"""
Test Attention layer against HuggingFace implementation.
"""

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeRotaryEmbedding,
)

from nanoqwen.layers import Attention, Rope
from tests.utils import ATOL, RTOL, copy_weights


def make_causal_mask(seq_len: int, batch_size: int) -> torch.Tensor:
    """Create a causal mask for HF attention (additive mask with -inf for masked positions)."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.float().masked_fill(mask, float("-inf"))
    # HF expects shape (batch, 1, seq, seq)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

# Weight mapping: HF param name -> your param name
# Qwen3 has q_norm/k_norm (RMSNorm on Q and K)
ATTENTION_MAPPING = {
    "q_proj.weight": "q_proj.weight",
    "k_proj.weight": "k_proj.weight",
    "v_proj.weight": "v_proj.weight",
    "o_proj.weight": "o_proj.weight",
    "q_norm.weight": "q_norm.weight",
    "k_norm.weight": "k_norm.weight",
}


def test_attention_forward(test_config, batch_size):
    """Test that Attention output matches HuggingFace."""
    seq_len = test_config.max_position_embeddings  # Always use max seq len for training

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, test_config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Create HF layers
    hf_attn = Qwen3MoeAttention(test_config, layer_idx=0)
    hf_rope = Qwen3MoeRotaryEmbedding(test_config)
    hf_attn.eval()

    # Create your layers
    nano_attn = Attention(test_config)
    nano_rope = Rope(test_config)
    nano_attn.eval()

    # Copy weights from HF to yours
    copy_weights(hf_attn, nano_attn, ATTENTION_MAPPING)

    # Create causal mask for HF (your impl has built-in causal mask)
    causal_mask = make_causal_mask(seq_len, batch_size)

    with torch.no_grad():
        # HF forward with causal mask
        hf_cos, hf_sin = hf_rope(x, position_ids)
        hf_out, _ = hf_attn(
            hidden_states=x,
            position_embeddings=(hf_cos, hf_sin),
            attention_mask=causal_mask,
        )

        # Your forward (has built-in causal mask)
        nano_sin, nano_cos = nano_rope(x)
        nano_out = nano_attn(x, nano_sin, nano_cos)

    # Compare outputs
    assert torch.allclose(hf_out, nano_out, rtol=RTOL, atol=ATOL), \
        f"Attention output mismatch. Max diff: {(hf_out - nano_out).abs().max()}"


def test_attention_output_shape(test_config, batch_size):
    """Test attention output has correct shape."""
    seq_len = test_config.max_position_embeddings

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, test_config.hidden_size)

    nano_attn = Attention(test_config)
    nano_rope = Rope(test_config)
    nano_attn.eval()

    with torch.no_grad():
        nano_sin, nano_cos = nano_rope(x)
        nano_out = nano_attn(x, nano_sin, nano_cos)

    assert nano_out.shape == x.shape


def test_attention_gqa(test_config, batch_size):
    """Test that GQA (grouped query attention) works correctly."""
    # test_config has num_attention_heads=8, num_key_value_heads=4
    # So we have 2 query heads per KV head
    seq_len = test_config.max_position_embeddings

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, test_config.hidden_size)

    hf_attn = Qwen3MoeAttention(test_config, layer_idx=0)
    hf_rope = Qwen3MoeRotaryEmbedding(test_config)
    hf_attn.eval()

    nano_attn = Attention(test_config)
    nano_rope = Rope(test_config)
    nano_attn.eval()

    copy_weights(hf_attn, nano_attn, ATTENTION_MAPPING)

    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    causal_mask = make_causal_mask(seq_len, batch_size)

    with torch.no_grad():
        hf_cos, hf_sin = hf_rope(x, position_ids)
        hf_out, _ = hf_attn(
            hidden_states=x,
            position_embeddings=(hf_cos, hf_sin),
            attention_mask=causal_mask,
        )

        nano_sin, nano_cos = nano_rope(x)
        nano_out = nano_attn(x, nano_sin, nano_cos)

    assert nano_out.shape == hf_out.shape, "GQA output shape mismatch"
    assert torch.allclose(hf_out, nano_out, rtol=RTOL, atol=ATOL), \
        f"GQA output mismatch. Max diff: {(hf_out - nano_out).abs().max()}"


def test_attention_gradient(test_config, batch_size):
    """Test that gradients flow correctly through your Attention."""
    seq_len = test_config.max_position_embeddings

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, test_config.hidden_size, requires_grad=True)

    nano_attn = Attention(test_config)
    nano_rope = Rope(test_config)

    nano_sin, nano_cos = nano_rope(x)
    out = nano_attn(x, nano_sin, nano_cos)
    out.sum().backward()

    assert x.grad is not None, "Gradient should flow through Attention"
