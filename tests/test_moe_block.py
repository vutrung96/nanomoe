"""
Test SparseMoeBlock against HuggingFace implementation.
"""

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from tests.utils import ATOL, RTOL, copy_weights

# TODO: Import your SparseMoeBlock implementation
# from moe import SparseMoeBlock


def build_moe_mapping(num_experts: int) -> dict[str, str]:
    """
    Build weight mapping for MoE block.

    This creates mappings for:
    - gate (router)
    - each expert's gate_proj, up_proj, down_proj

    Note: Qwen3 MoE does NOT have shared_expert (unlike Qwen2 MoE)
    """
    mapping = {
        # Router
        "gate.weight": "gate.weight",
    }

    # Add mappings for each expert
    for i in range(num_experts):
        mapping[f"experts.{i}.gate_proj.weight"] = f"experts.{i}.gate_proj.weight"
        mapping[f"experts.{i}.up_proj.weight"] = f"experts.{i}.up_proj.weight"
        mapping[f"experts.{i}.down_proj.weight"] = f"experts.{i}.down_proj.weight"

    return mapping


def test_moe_block_forward(test_config, random_hidden_states):
    """Test that SparseMoeBlock output matches HuggingFace."""
    # Create HF block
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)
    hf_moe.eval()

    # TODO: Create your block
    # your_moe = SparseMoeBlock(test_config)
    # your_moe.eval()

    # Build and apply weight mapping
    mapping = build_moe_mapping(test_config.num_experts)
    # copy_weights(hf_moe, your_moe, mapping)

    # Forward pass
    with torch.no_grad():
        hf_out, hf_router_logits = hf_moe(random_hidden_states)
        # your_out, your_router_logits = your_moe(random_hidden_states)

    # Compare outputs
    # assert torch.allclose(hf_out, your_out, rtol=RTOL, atol=ATOL), \
    #     f"MoE output mismatch. Max diff: {(hf_out - your_out).abs().max()}"
    # assert torch.allclose(hf_router_logits, your_router_logits, rtol=RTOL, atol=ATOL), \
    #     f"Router logits mismatch. Max diff: {(hf_router_logits - your_router_logits).abs().max()}"

    # Placeholder
    assert hf_out.shape == random_hidden_states.shape, "MoE should preserve shape"
    batch, seq, _ = random_hidden_states.shape
    assert hf_router_logits.shape == (batch * seq, test_config.num_experts), \
        f"Router logits shape: {hf_router_logits.shape}"


def test_moe_router_logits(test_config, random_hidden_states):
    """Test that router produces valid logits."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)
    hf_moe.eval()

    with torch.no_grad():
        _, router_logits = hf_moe(random_hidden_states)

    # Router logits should be finite
    assert torch.isfinite(router_logits).all(), "Router logits should be finite"

    # Shape check
    batch, seq, _ = random_hidden_states.shape
    assert router_logits.shape == (batch * seq, test_config.num_experts)


def test_moe_expert_selection(test_config, random_hidden_states):
    """Test that top-k experts are selected correctly."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)
    hf_moe.eval()

    with torch.no_grad():
        _, router_logits = hf_moe(random_hidden_states)

    # Compute top-k selection
    routing_weights = torch.softmax(router_logits, dim=-1)
    top_weights, top_indices = torch.topk(
        routing_weights, test_config.num_experts_per_tok, dim=-1
    )

    # Should select exactly num_experts_per_tok experts per token
    assert top_indices.shape[-1] == test_config.num_experts_per_tok

    # Weights should be positive
    assert (top_weights > 0).all(), "Top-k weights should be positive"


def test_moe_experts_structure(test_config, random_hidden_states):
    """Test that MoE experts are structured correctly."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)

    # Qwen3 MoE has num_experts expert MLPs (no shared expert)
    assert len(hf_moe.experts) == test_config.num_experts

    # Each expert should be an MLP
    with torch.no_grad():
        batch, seq, hidden = random_hidden_states.shape
        flat_hidden = random_hidden_states.view(-1, hidden)

        # Test that each expert can process the input
        for i, expert in enumerate(hf_moe.experts):
            expert_out = expert(flat_hidden)
            assert expert_out.shape == flat_hidden.shape, f"Expert {i} output shape mismatch"


def test_moe_gradient(test_config, random_hidden_states):
    """Test that gradients flow correctly through MoE block."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)

    x = random_hidden_states.clone().requires_grad_(True)
    out, router_logits = hf_moe(x)
    (out.sum() + router_logits.sum()).backward()

    assert x.grad is not None, "Gradient should flow through MoE"
    assert hf_moe.gate.weight.grad is not None, "Router should have gradient"
