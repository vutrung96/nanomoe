from dataclasses import dataclass


@dataclass
class MoeConfig:
    """Generic MoE model configuration.

    This config is independent of any specific framework (HuggingFace, etc.)
    and contains only the fields actually used by the nanomoe implementation.
    """

    # Core dimensions
    hidden_size: int

    # Attention
    num_attention_heads: int
    num_key_value_heads: int

    # Layers
    num_layers: int

    # Vocab
    vocab_size: int
    tied_embedding: bool

    # RoPE
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048

    # MoE
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 1024
    norm_topk_prob: bool = True

    # Normalization
    rms_norm_eps: float = 1e-6
