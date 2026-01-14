"""
Shared fixtures for nanomoe testing harness.
"""

import pytest
import torch

from nanomoe.config import MoeConfig
from tests.config_utils import moe_config_to_qwen3


@pytest.fixture
def moe_config():
    """Generic MoeConfig for testing."""
    return MoeConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,  # GQA: 8/4 = 2 heads per KV group
        num_layers=2,
        vocab_size=1000,
        tied_embedding=False,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        norm_topk_prob=True,
    )


@pytest.fixture
def test_config(moe_config):
    """HuggingFace Qwen3MoeConfig for comparison tests.

    This fixture provides backwards compatibility - tests that need
    HuggingFace models should use this fixture.
    """
    return moe_config_to_qwen3(moe_config)


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 128


@pytest.fixture
def random_hidden_states(moe_config, batch_size, seq_len):
    """Random input tensor of shape (batch, seq, hidden_size)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, moe_config.hidden_size)


@pytest.fixture
def position_ids(batch_size, seq_len):
    """Position IDs for RoPE."""
    return torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
