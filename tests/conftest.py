"""
Shared fixtures for Qwen3 MoE testing harness.
"""

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


@pytest.fixture
def test_config():
    """Minimal config for fast testing."""
    config = Qwen3MoeConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,  # GQA: 8/4 = 2 heads per KV group
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=256,
        vocab_size=1000,
        hidden_act="si lu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,  # Qwen3 has no attention bias by default
        norm_topk_prob=False,
    )
    config._attn_implementation = "eager"
    return config


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 128


@pytest.fixture
def random_hidden_states(test_config, batch_size, seq_len):
    """Random input tensor of shape (batch, seq, hidden_size)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, test_config.hidden_size)


@pytest.fixture
def position_ids(batch_size, seq_len):
    """Position IDs for RoPE."""
    return torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
