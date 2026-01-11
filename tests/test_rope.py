"""
Test Rotary Position Embeddings against HuggingFace implementation.
"""

import torch
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    apply_rotary_pos_emb,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeRotaryEmbedding,
)

from nanoqwen.layers import Rope, apply_rope
from tests.utils import ATOL, RTOL


def test_rotary_embedding_cos_sin(test_config, batch_size, seq_len):
    """Test that rotary embedding produces matching cos/sin."""
    hf_rope = Qwen3MoeRotaryEmbedding(test_config)
    nano_rope = Rope(test_config)

    # Create dummy input and position_ids
    x = torch.randn(batch_size, seq_len, test_config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        hf_cos, hf_sin = hf_rope(x, position_ids)
        nano_sin, nano_cos = nano_rope(x)

    # Check shapes match
    head_dim = test_config.hidden_size // test_config.num_attention_heads
    assert hf_cos.shape == (batch_size, seq_len, head_dim), (
        f"HF cos shape: {hf_cos.shape}"
    )
    assert hf_sin.shape == (batch_size, seq_len, head_dim), (
        f"HF sin shape: {hf_sin.shape}"
    )

    # Your implementation returns (1, 1, max_seq, head_dim) - squeeze and slice
    nano_cos_slice = nano_cos.squeeze(0).squeeze(0)[:seq_len, :]
    nano_sin_slice = nano_sin.squeeze(0).squeeze(0)[:seq_len, :]

    # HF returns (batch, seq, head_dim) - compare with one batch
    assert nano_cos_slice.shape == hf_cos[0].shape, (
        f"Nano cos shape {nano_cos_slice.shape} != HF cos shape {hf_cos[0].shape}"
    )

    # Compare values (HF broadcasts same values across batch)
    assert torch.allclose(hf_cos[0], nano_cos_slice, rtol=RTOL, atol=ATOL), (
        f"RoPE cos mismatch. Max diff: {(hf_cos[0] - nano_cos_slice).abs().max()}"
    )
    assert torch.allclose(hf_sin[0], nano_sin_slice, rtol=RTOL, atol=ATOL), (
        f"RoPE sin mismatch. Max diff: {(hf_sin[0] - nano_sin_slice).abs().max()}"
    )


def test_apply_rope_vs_hf(test_config, batch_size, seq_len):
    """Test your apply_rope matches HuggingFace's apply_rotary_pos_emb."""
    hf_rope = Qwen3MoeRotaryEmbedding(test_config)
    nano_rope = Rope(test_config)

    head_dim = test_config.hidden_size // test_config.num_attention_heads

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, head_dim)

    # For HF: need (batch, heads, seq, head_dim) and will use 1 head for simplicity
    x_hf = x.unsqueeze(1)  # (batch, 1, seq, head_dim)

    x_dummy = torch.randn(batch_size, seq_len, test_config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        # HF
        hf_cos, hf_sin = hf_rope(x_dummy, position_ids)
        hf_out, _ = apply_rotary_pos_emb(x_hf, x_hf, hf_cos, hf_sin)
        hf_out = hf_out.squeeze(1)  # back to (batch, seq, head_dim)

        # Yours - squeeze to (seq, head_dim) then slice
        nano_sin, nano_cos = nano_rope(x)
        nano_sin_slice = nano_sin.squeeze(0).squeeze(0)[:seq_len, :]
        nano_cos_slice = nano_cos.squeeze(0).squeeze(0)[:seq_len, :]
        nano_out = apply_rope(x, nano_sin_slice, nano_cos_slice)

    assert torch.allclose(hf_out, nano_out, rtol=RTOL, atol=ATOL), (
        f"apply_rope mismatch. Max diff: {(hf_out - nano_out).abs().max()}"
    )
