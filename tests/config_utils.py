"""
Utilities for converting between MoeConfig and HuggingFace Qwen3MoeConfig.

This allows tests to:
1. Create a generic MoeConfig
2. Convert it to Qwen3MoeConfig for HuggingFace reference implementations
3. Compare outputs between nanomoe and HuggingFace
"""

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nanomoe.config import MoeConfig


def moe_config_to_qwen3(config: MoeConfig) -> Qwen3MoeConfig:
    """Convert generic MoeConfig to HuggingFace Qwen3MoeConfig.

    Sets sensible defaults for fields not present in MoeConfig
    that are required by HuggingFace models.
    """
    qwen_config = Qwen3MoeConfig(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        rope_theta=config.rope_theta,
        max_position_embeddings=config.max_position_embeddings,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        moe_intermediate_size=config.moe_intermediate_size,
        norm_topk_prob=config.norm_topk_prob,
        rms_norm_eps=config.rms_norm_eps,
        num_hidden_layers=config.num_layers,
        vocab_size=config.vocab_size,
        # HF-required defaults
        intermediate_size=config.hidden_size * 4,
        shared_expert_intermediate_size=config.hidden_size,
        hidden_act="silu",
        attention_bias=False,
    )
    qwen_config._attn_implementation = "eager"
    return qwen_config


def qwen3_to_moe_config(qwen_config: Qwen3MoeConfig) -> MoeConfig:
    """Convert HuggingFace Qwen3MoeConfig to generic MoeConfig.

    Useful for loading existing HF configs into nanomoe.
    """
    return MoeConfig(
        hidden_size=qwen_config.hidden_size,
        num_attention_heads=qwen_config.num_attention_heads,
        num_key_value_heads=qwen_config.num_key_value_heads,
        num_layers=qwen_config.num_hidden_layers,
        vocab_size=qwen_config.vocab_size,
        tied_embedding=False,  # HF doesn't have this field
        rope_theta=qwen_config.rope_theta,
        max_position_embeddings=qwen_config.max_position_embeddings,
        num_experts=qwen_config.num_experts,
        num_experts_per_tok=qwen_config.num_experts_per_tok,
        moe_intermediate_size=qwen_config.moe_intermediate_size,
        norm_topk_prob=qwen_config.norm_topk_prob,
        rms_norm_eps=qwen_config.rms_norm_eps,
    )
