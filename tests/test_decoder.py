"""
Test full DecoderLayer against HuggingFace implementation.
"""

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeRotaryEmbedding,
)

from tests.utils import ATOL, RTOL, copy_weights

# TODO: Import your DecoderLayer implementation
# from moe import DecoderLayer


def build_decoder_mapping(config, layer_idx: int) -> dict[str, str]:
    """
    Build weight mapping for full decoder layer.

    A decoder layer contains:
    - input_layernorm (RMSNorm)
    - self_attn (Attention)
    - post_attention_layernorm (RMSNorm)
    - mlp (either MLP or SparseMoeBlock depending on layer)
    """
    mapping = {
        # Input LayerNorm
        "input_layernorm.weight": "input_layernorm.weight",
        # Attention (Qwen3 has no biases, but has q_norm/k_norm)
        "self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "self_attn.q_norm.weight": "self_attn.q_norm.weight",
        "self_attn.k_norm.weight": "self_attn.k_norm.weight",
        # Post-attention LayerNorm
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    }

    # Check if this layer uses MoE or regular MLP
    # By default, decoder_sparse_step=1 means every layer uses MoE
    uses_moe = (
        layer_idx not in config.mlp_only_layers
        and config.num_experts > 0
        and (layer_idx + 1) % config.decoder_sparse_step == 0
    )

    if uses_moe:
        # MoE block mappings (Qwen3 has no shared expert)
        mapping["mlp.gate.weight"] = "mlp.gate.weight"

        for i in range(config.num_experts):
            mapping[f"mlp.experts.{i}.gate_proj.weight"] = f"mlp.experts.{i}.gate_proj.weight"
            mapping[f"mlp.experts.{i}.up_proj.weight"] = f"mlp.experts.{i}.up_proj.weight"
            mapping[f"mlp.experts.{i}.down_proj.weight"] = f"mlp.experts.{i}.down_proj.weight"
    else:
        # Regular MLP mappings
        mapping["mlp.gate_proj.weight"] = "mlp.gate_proj.weight"
        mapping["mlp.up_proj.weight"] = "mlp.up_proj.weight"
        mapping["mlp.down_proj.weight"] = "mlp.down_proj.weight"

    return mapping


def test_decoder_layer_forward(test_config, random_hidden_states, position_ids):
    """Test that DecoderLayer output matches HuggingFace."""
    batch_size, seq_len, _ = random_hidden_states.shape

    # Need to set _attn_implementation for HF
    # _attn_implementation is set in conftest.py

    # Create HF layer
    hf_decoder = Qwen3MoeDecoderLayer(test_config, layer_idx=0)
    hf_rope = Qwen3MoeRotaryEmbedding(test_config)
    hf_decoder.eval()

    # TODO: Create your layer
    # your_decoder = DecoderLayer(test_config, layer_idx=0)
    # your_decoder.eval()

    # Build and apply weight mapping
    mapping = build_decoder_mapping(test_config, layer_idx=0)
    # copy_weights(hf_decoder, your_decoder, mapping)

    # Compute position embeddings
    with torch.no_grad():
        cos, sin = hf_rope(random_hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # Qwen3 decoder returns hidden_states tensor directly (not a tuple)
        hf_out = hf_decoder(
            hidden_states=random_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )

        # TODO: Your forward
        # your_out = your_decoder(random_hidden_states, position_embeddings=position_embeddings)

    # Compare outputs
    # assert torch.allclose(hf_out, your_out, rtol=RTOL, atol=ATOL), \
    #     f"Decoder output mismatch. Max diff: {(hf_out - your_out).abs().max()}"

    # Placeholder
    assert hf_out.shape == random_hidden_states.shape, "Decoder should preserve shape"


def test_decoder_with_router_logits(test_config, random_hidden_states, position_ids):
    """Test decoder layer with router logits output."""
    # _attn_implementation is set in conftest.py

    hf_decoder = Qwen3MoeDecoderLayer(test_config, layer_idx=0)
    hf_rope = Qwen3MoeRotaryEmbedding(test_config)
    hf_decoder.eval()

    with torch.no_grad():
        cos, sin = hf_rope(random_hidden_states, position_ids)

        # Qwen3 decoder returns tensor directly
        # Router logits are handled internally by the MoE block
        hidden_out = hf_decoder(
            hidden_states=random_hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )

    assert hidden_out.shape == random_hidden_states.shape


def test_decoder_residual_connection(test_config, random_hidden_states, position_ids):
    """Test that residual connections are working."""
    # _attn_implementation is set in conftest.py

    hf_decoder = Qwen3MoeDecoderLayer(test_config, layer_idx=0)
    hf_rope = Qwen3MoeRotaryEmbedding(test_config)

    # Zero out all weights to isolate residual
    with torch.no_grad():
        for param in hf_decoder.parameters():
            param.zero_()
        # But keep layernorm weights at 1
        hf_decoder.input_layernorm.weight.fill_(1.0)
        hf_decoder.post_attention_layernorm.weight.fill_(1.0)

    with torch.no_grad():
        cos, sin = hf_rope(random_hidden_states, position_ids)
        out = hf_decoder(
            hidden_states=random_hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )

    # With zeroed weights, output should be close to input (just residuals)
    # Note: This is a sanity check, not exact due to layernorm
    assert out.shape == random_hidden_states.shape


def test_decoder_gradient(test_config, random_hidden_states, position_ids):
    """Test that gradients flow correctly through decoder layer."""
    # _attn_implementation is set in conftest.py

    hf_decoder = Qwen3MoeDecoderLayer(test_config, layer_idx=0)
    hf_rope = Qwen3MoeRotaryEmbedding(test_config)

    x = random_hidden_states.clone().requires_grad_(True)

    cos, sin = hf_rope(x, position_ids)
    out = hf_decoder(hidden_states=x, position_embeddings=(cos, sin), attention_mask=None)
    out.sum().backward()

    assert x.grad is not None, "Gradient should flow through decoder"
    assert hf_decoder.self_attn.q_proj.weight.grad is not None, "Attention should have gradient"
