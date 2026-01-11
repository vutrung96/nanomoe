"""
Test RMSNorm layer against HuggingFace implementation.
"""

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from tests.utils import ATOL, RTOL, copy_weights

# TODO: Import your RMSNorm implementation
# from moe import RMSNorm

# Weight mapping: HF param name -> your param name
RMSNORM_MAPPING = {
    "weight": "weight",  # Adjust if your param name differs
}


def test_rmsnorm_forward(test_config, random_hidden_states):
    """Test that RMSNorm output matches HuggingFace."""
    # Create HF layer
    hf_norm = Qwen3MoeRMSNorm(test_config.hidden_size, eps=test_config.rms_norm_eps)
    hf_norm.eval()

    # TODO: Create your layer
    # your_norm = RMSNorm(test_config.hidden_size, eps=test_config.rms_norm_eps)
    # your_norm.eval()

    # Copy weights
    # copy_weights(hf_norm, your_norm, RMSNORM_MAPPING)

    # Forward pass
    with torch.no_grad():
        hf_out = hf_norm(random_hidden_states)
        # your_out = your_norm(random_hidden_states)

    # Compare outputs
    # assert torch.allclose(hf_out, your_out, rtol=RTOL, atol=ATOL), \
    #     f"RMSNorm output mismatch. Max diff: {(hf_out - your_out).abs().max()}"

    # Placeholder assertion until you implement your layer
    assert hf_out.shape == random_hidden_states.shape, "HF RMSNorm shape check"


def test_rmsnorm_gradient(test_config, random_hidden_states):
    """Test that gradients flow correctly through RMSNorm."""
    hf_norm = Qwen3MoeRMSNorm(test_config.hidden_size, eps=test_config.rms_norm_eps)

    # TODO: Create your layer and test gradients match
    # your_norm = RMSNorm(test_config.hidden_size, eps=test_config.rms_norm_eps)
    # copy_weights(hf_norm, your_norm, RMSNORM_MAPPING)

    x = random_hidden_states.clone().requires_grad_(True)
    hf_out = hf_norm(x)
    hf_out.sum().backward()

    # Placeholder
    assert x.grad is not None, "Gradient should flow through RMSNorm"
