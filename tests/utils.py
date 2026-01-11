"""
Shared utilities for Qwen3 MoE testing harness.
"""

import torch


def copy_weights(hf_module, your_module, mapping: dict[str, str]):
    """
    Copy weights from HF module to your module using explicit mapping.

    Args:
        hf_module: HuggingFace module
        your_module: Your from-scratch module
        mapping: Dict mapping HF param names -> your param names
                 e.g. {"gate_proj.weight": "w_gate.weight", ...}
    """
    hf_state = hf_module.state_dict()
    your_state = your_module.state_dict()

    for hf_name, your_name in mapping.items():
        if hf_name in hf_state:
            your_state[your_name] = hf_state[hf_name]

    your_module.load_state_dict(your_state)


# Tolerances for comparisons
RTOL = 1e-4
ATOL = 1e-5
