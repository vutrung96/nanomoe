"""
Integration test for full Qwen3 MoE model end-to-end.
"""

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeModel,
    Qwen3MoeForCausalLM,
)

from nanomoe.model import Model
from tests.utils import (
    copy_weights,
    copy_expert_weights,
    make_causal_mask,
    make_causal_mask_bool,
)


def build_model_mapping(config) -> dict[str, str]:
    """
    Build weight mapping for the full Qwen3 MoE model.

    Structure:
    - embed_tokens: token embeddings
    - layers: list of decoder layers
    - norm: final RMSNorm
    """
    mapping = {
        # Embeddings
        "embed_tokens.weight": "embed_tokens.weight",
        # Final norm
        "norm.weight": "norm.weight",
    }

    # Add mappings for each layer
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"

        # LayerNorms
        mapping[f"{prefix}.input_layernorm.weight"] = f"{prefix}.input_layernorm.weight"
        mapping[f"{prefix}.post_attention_layernorm.weight"] = f"{prefix}.post_attention_layernorm.weight"

        # Attention (Qwen3 has no biases, has q_norm/k_norm)
        mapping[f"{prefix}.self_attn.q_proj.weight"] = f"{prefix}.self_attn.q_proj.weight"
        mapping[f"{prefix}.self_attn.k_proj.weight"] = f"{prefix}.self_attn.k_proj.weight"
        mapping[f"{prefix}.self_attn.v_proj.weight"] = f"{prefix}.self_attn.v_proj.weight"
        mapping[f"{prefix}.self_attn.o_proj.weight"] = f"{prefix}.self_attn.o_proj.weight"
        mapping[f"{prefix}.self_attn.q_norm.weight"] = f"{prefix}.self_attn.q_norm.weight"
        mapping[f"{prefix}.self_attn.k_norm.weight"] = f"{prefix}.self_attn.k_norm.weight"

        # Check if this layer uses MoE or regular MLP
        uses_moe = (
            layer_idx not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )

        if uses_moe:
            # MoE block (Qwen3 has no shared expert)
            mapping[f"{prefix}.mlp.gate.weight"] = f"{prefix}.mlp.gate.weight"
            for i in range(config.num_experts):
                mapping[f"{prefix}.mlp.experts.{i}.gate_proj.weight"] = f"{prefix}.mlp.experts.{i}.gate_proj.weight"
                mapping[f"{prefix}.mlp.experts.{i}.up_proj.weight"] = f"{prefix}.mlp.experts.{i}.up_proj.weight"
                mapping[f"{prefix}.mlp.experts.{i}.down_proj.weight"] = f"{prefix}.mlp.experts.{i}.down_proj.weight"
        else:
            # Regular MLP
            mapping[f"{prefix}.mlp.gate_proj.weight"] = f"{prefix}.mlp.gate_proj.weight"
            mapping[f"{prefix}.mlp.up_proj.weight"] = f"{prefix}.mlp.up_proj.weight"
            mapping[f"{prefix}.mlp.down_proj.weight"] = f"{prefix}.mlp.down_proj.weight"

    return mapping


def build_causal_lm_mapping(config) -> dict[str, str]:
    """Build weight mapping for Qwen3MoeForCausalLM (model + lm_head)."""
    # Get base model mappings with 'model.' prefix
    base_mapping = build_model_mapping(config)
    mapping = {f"model.{k}": f"model.{v}" for k, v in base_mapping.items()}

    # Add lm_head
    mapping["lm_head.weight"] = "lm_head.weight"

    return mapping


def test_model_forward(test_config):
    """Test full model forward pass with token inputs."""
    batch_size, seq_len = 2, 16

    # Create random token IDs
    torch.manual_seed(42)
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

    # Create HF model
    hf_model = Qwen3MoeModel(test_config)
    hf_model.eval()

    # TODO: Create your model
    # your_model = YourQwen3MoeModel(test_config)
    # your_model.eval()

    # Copy weights
    # mapping = build_model_mapping(test_config)
    # copy_weights(hf_model, your_model, mapping)

    with torch.no_grad():
        hf_outputs = hf_model(input_ids=input_ids)
        hf_hidden = hf_outputs.last_hidden_state

        # TODO: Your forward
        # your_hidden = your_model(input_ids)

    # Check output shape
    assert hf_hidden.shape == (batch_size, seq_len, test_config.hidden_size), \
        f"Model output shape: {hf_hidden.shape}"

    # Compare outputs
    # assert torch.allclose(hf_hidden, your_hidden, rtol=RTOL, atol=ATOL), \
    #     f"Model output mismatch. Max diff: {(hf_hidden - your_hidden).abs().max()}"


def test_causal_lm_forward(test_config):
    """Test CausalLM forward pass (model + lm_head)."""
    batch_size, seq_len = 2, 16

    torch.manual_seed(42)
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

    # Create HF model
    hf_model = Qwen3MoeForCausalLM(test_config)
    hf_model.eval()

    with torch.no_grad():
        hf_outputs = hf_model(input_ids=input_ids)
        hf_logits = hf_outputs.logits

    # Check output shape: (batch, seq, vocab_size)
    assert hf_logits.shape == (batch_size, seq_len, test_config.vocab_size), \
        f"Logits shape: {hf_logits.shape}"


def test_causal_lm_loss(test_config):
    """Test CausalLM with language modeling loss."""
    batch_size, seq_len = 2, 16

    torch.manual_seed(42)
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
    # Labels are shifted input_ids (standard LM setup)
    labels = input_ids.clone()

    hf_model = Qwen3MoeForCausalLM(test_config)
    hf_model.eval()

    with torch.no_grad():
        hf_outputs = hf_model(input_ids=input_ids, labels=labels)

    # Should have loss
    assert hf_outputs.loss is not None, "Should compute loss when labels provided"
    assert hf_outputs.loss.shape == (), "Loss should be scalar"
    assert torch.isfinite(hf_outputs.loss), "Loss should be finite"


def test_model_gradient(test_config):
    """Test that gradients flow through the full model."""
    batch_size, seq_len = 2, 8

    torch.manual_seed(42)
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    hf_model = Qwen3MoeForCausalLM(test_config)

    outputs = hf_model(input_ids=input_ids, labels=labels)
    outputs.loss.backward()

    # Check gradients exist
    assert hf_model.model.embed_tokens.weight.grad is not None, \
        "Embedding should have gradient"
    assert hf_model.model.layers[0].self_attn.q_proj.weight.grad is not None, \
        "First layer attention should have gradient"
    assert hf_model.lm_head.weight.grad is not None, \
        "LM head should have gradient"


def test_model_with_attention_mask(test_config):
    """Test model with attention mask (padding)."""
    batch_size, seq_len = 2, 16

    torch.manual_seed(42)
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

    # Create attention mask with some padding
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, -4:] = 0  # Mask last 4 tokens of first sequence
    attention_mask[1, -2:] = 0  # Mask last 2 tokens of second sequence

    hf_model = Qwen3MoeModel(test_config)
    hf_model.eval()

    with torch.no_grad():
        hf_outputs = hf_model(input_ids=input_ids, attention_mask=attention_mask)
        hf_hidden = hf_outputs.last_hidden_state

    assert hf_hidden.shape == (batch_size, seq_len, test_config.hidden_size)


def test_model_incremental_decoding(test_config):
    """Test model with KV cache for incremental decoding."""
    batch_size = 2
    prompt_len = 8
    gen_len = 4

    torch.manual_seed(42)
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, prompt_len))

    hf_model = Qwen3MoeForCausalLM(test_config)
    hf_model.eval()

    with torch.no_grad():
        # First forward: process prompt, get cache
        outputs = hf_model(input_ids=input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        # Greedy decode next token
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        # Incremental decoding steps
        for _ in range(gen_len - 1):
            outputs = hf_model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Verify cache structure
    assert past_kv is not None, "Should return KV cache"
    # Cache should have entries for each layer
    assert len(past_kv) == test_config.num_hidden_layers


def test_model_deterministic(test_config):
    """Test that model output is deterministic with same input."""
    batch_size, seq_len = 2, 16

    torch.manual_seed(42)
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

    hf_model = Qwen3MoeModel(test_config)
    hf_model.eval()

    with torch.no_grad():
        out1 = hf_model(input_ids=input_ids).last_hidden_state
        out2 = hf_model(input_ids=input_ids).last_hidden_state

    assert torch.allclose(out1, out2), "Model should be deterministic in eval mode"


def test_model_router_logits(test_config):
    """
    Test that router logits are returned when requested.

    This is useful for computing auxiliary load balancing loss during training.
    """
    batch_size, seq_len = 2, 16

    torch.manual_seed(42)
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

    hf_model = Qwen3MoeForCausalLM(test_config)
    hf_model.eval()

    with torch.no_grad():
        outputs = hf_model(input_ids=input_ids, output_router_logits=True)

    # Should have router_logits in output
    assert outputs.router_logits is not None, "Should return router logits"

    # Router logits is a tuple with one tensor per MoE layer
    num_moe_layers = sum(
        1 for i in range(test_config.num_hidden_layers)
        if i not in test_config.mlp_only_layers
        and test_config.num_experts > 0
        and (i + 1) % test_config.decoder_sparse_step == 0
    )

    assert len(outputs.router_logits) == num_moe_layers, \
        f"Expected {num_moe_layers} router logit tensors, got {len(outputs.router_logits)}"

    # Each router logit should have shape (batch * seq, num_experts)
    for i, router_logit in enumerate(outputs.router_logits):
        expected_shape = (batch_size * seq_len, test_config.num_experts)
        assert router_logit.shape == expected_shape, \
            f"Router logit {i} shape: {router_logit.shape}, expected {expected_shape}"


# =============================================================================
# User Model tests
# =============================================================================


def test_user_model_forward_smoke(moe_config):
    """Smoke test: Model can run forward pass."""
    model = Model(moe_config)
    model.eval()

    batch_size, seq_len = 2, 16
    torch.manual_seed(42)
    input_ids = torch.randint(0, moe_config.vocab_size, (batch_size, seq_len))
    mask = make_causal_mask_bool(seq_len)

    with torch.no_grad():
        logits = model(input_ids, mask)

    assert logits.shape == (batch_size, seq_len, moe_config.vocab_size)


def test_user_model_gradient_smoke(moe_config):
    """Smoke test: gradients flow through Model."""
    model = Model(moe_config)

    batch_size, seq_len = 2, 8
    torch.manual_seed(42)
    input_ids = torch.randint(0, moe_config.vocab_size, (batch_size, seq_len))
    mask = make_causal_mask_bool(seq_len)

    logits = model(input_ids, mask)
    logits.sum().backward()

    assert model.embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert model.blocks[0].attention.q_proj.weight.grad is not None


def build_user_model_mapping(config) -> dict[str, str]:
    """
    Build weight mapping from HF Qwen3MoeForCausalLM to user's Model.

    HuggingFace -> User Model naming:
    - model.embed_tokens -> embedding
    - model.layers[i].input_layernorm -> blocks[i].norm1
    - model.layers[i].self_attn -> blocks[i].attention
    - model.layers[i].post_attention_layernorm -> blocks[i].norm2
    - model.layers[i].mlp.gate -> blocks[i].moe.router.gate
    - model.norm -> rms_f
    - lm_head -> lm_head
    """
    mapping = {
        # Embeddings
        "model.embed_tokens.weight": "embedding.weight",
        # Final norm
        "model.norm.weight": "rms_f.weight",
        # LM head
        "lm_head.weight": "lm_head.weight",
    }

    # Add mappings for each layer
    for layer_idx in range(config.num_hidden_layers):
        hf_prefix = f"model.layers.{layer_idx}"
        user_prefix = f"blocks.{layer_idx}"

        # LayerNorms
        mapping[f"{hf_prefix}.input_layernorm.weight"] = f"{user_prefix}.norm1.weight"
        mapping[f"{hf_prefix}.post_attention_layernorm.weight"] = f"{user_prefix}.norm2.weight"

        # Attention
        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = f"{user_prefix}.attention.q_proj.weight"
        mapping[f"{hf_prefix}.self_attn.k_proj.weight"] = f"{user_prefix}.attention.k_proj.weight"
        mapping[f"{hf_prefix}.self_attn.v_proj.weight"] = f"{user_prefix}.attention.v_proj.weight"
        mapping[f"{hf_prefix}.self_attn.o_proj.weight"] = f"{user_prefix}.attention.o_proj.weight"
        mapping[f"{hf_prefix}.self_attn.q_norm.weight"] = f"{user_prefix}.attention.q_norm.weight"
        mapping[f"{hf_prefix}.self_attn.k_norm.weight"] = f"{user_prefix}.attention.k_norm.weight"

        # Router
        mapping[f"{hf_prefix}.mlp.gate.weight"] = f"{user_prefix}.moe.router.gate.weight"

    return mapping


def test_user_model_vs_hf(test_config, moe_config):
    """Test user Model output matches HuggingFace Qwen3MoeForCausalLM."""
    batch_size, seq_len = 2, 16

    torch.manual_seed(42)
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

    # Create HF model
    hf_model = Qwen3MoeForCausalLM(test_config)
    hf_model.eval()

    # Create user model
    user_model = Model(moe_config)
    user_model.eval()

    # Copy weights
    mapping = build_user_model_mapping(test_config)
    copy_weights(hf_model, user_model, mapping)

    # Copy expert weights for each layer
    for layer_idx in range(test_config.num_hidden_layers):
        hf_moe = hf_model.model.layers[layer_idx].mlp
        user_experts = user_model.blocks[layer_idx].moe.experts
        copy_expert_weights(hf_moe, user_experts, test_config.num_experts)

    # Create masks
    hf_mask = make_causal_mask(seq_len, batch_size)
    user_mask = make_causal_mask_bool(seq_len)

    with torch.no_grad():
        hf_outputs = hf_model(input_ids=input_ids, attention_mask=hf_mask)
        hf_logits = hf_outputs.logits

        user_logits = user_model(input_ids, user_mask)

    # Use relaxed tolerances for full model - errors accumulate across layers
    model_atol = 1e-3
    model_rtol = 1e-3
    max_diff = (hf_logits - user_logits).abs().max().item()
    assert torch.allclose(hf_logits, user_logits, rtol=model_rtol, atol=model_atol), \
        f"Model output mismatch. Max diff: {max_diff}"
