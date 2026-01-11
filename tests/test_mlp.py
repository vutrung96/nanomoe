"""
Test MLP (feed-forward) layer against HuggingFace implementation.
"""

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

from tests.utils import ATOL, RTOL, copy_weights

# TODO: Import your MLP implementation
# from moe import MLP

# Weight mapping: HF param name -> your param name
MLP_MAPPING = {
    "gate_proj.weight": "gate_proj.weight",  # Adjust to match your naming
    "up_proj.weight": "up_proj.weight",
    "down_proj.weight": "down_proj.weight",
}


def test_mlp_forward(test_config, random_hidden_states):
    """Test that MLP output matches HuggingFace."""
    # Create HF layer with intermediate_size
    hf_mlp = Qwen3MoeMLP(test_config, intermediate_size=test_config.intermediate_size)
    hf_mlp.eval()

    # TODO: Create your layer
    # your_mlp = MLP(test_config, intermediate_size=test_config.intermediate_size)
    # your_mlp.eval()

    # Copy weights
    # copy_weights(hf_mlp, your_mlp, MLP_MAPPING)

    # Forward pass
    with torch.no_grad():
        hf_out = hf_mlp(random_hidden_states)
        # your_out = your_mlp(random_hidden_states)

    # Compare outputs
    # assert torch.allclose(hf_out, your_out, rtol=RTOL, atol=ATOL), \
    #     f"MLP output mismatch. Max diff: {(hf_out - your_out).abs().max()}"

    # Placeholder
    assert hf_out.shape == random_hidden_states.shape, "MLP should preserve shape"


def test_mlp_expert_size(test_config, random_hidden_states):
    """Test MLP with moe_intermediate_size (for expert MLPs)."""
    # Expert MLPs use moe_intermediate_size
    hf_expert_mlp = Qwen3MoeMLP(
        test_config, intermediate_size=test_config.moe_intermediate_size
    )
    hf_expert_mlp.eval()

    # TODO: Create your expert MLP
    # your_expert_mlp = MLP(test_config, intermediate_size=test_config.moe_intermediate_size)
    # copy_weights(hf_expert_mlp, your_expert_mlp, MLP_MAPPING)

    with torch.no_grad():
        hf_out = hf_expert_mlp(random_hidden_states)
        # your_out = your_expert_mlp(random_hidden_states)

    # assert torch.allclose(hf_out, your_out, rtol=RTOL, atol=ATOL)

    # Placeholder
    assert hf_out.shape == random_hidden_states.shape


def test_mlp_gradient(test_config, random_hidden_states):
    """Test that gradients flow correctly through MLP."""
    hf_mlp = Qwen3MoeMLP(test_config, intermediate_size=test_config.intermediate_size)

    x = random_hidden_states.clone().requires_grad_(True)
    out = hf_mlp(x)
    out.sum().backward()

    assert x.grad is not None, "Gradient should flow through MLP"
    assert hf_mlp.gate_proj.weight.grad is not None, "gate_proj should have gradient"
