import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.gate = nn.Linear(
            self.config.hidden_size, self.config.num_experts, bias=False
        )

    def forward(self, x):
        routing_probs = F.softmax(self.gate(x), dim=-1)
        top_k_probs, top_k_indices = torch.topk(
            routing_probs, self.config.num_experts_per_tok
        )
        routing_map = (
            torch.zeros_like(routing_probs).scatter_(-1, top_k_indices, True).bool()
        )
        routing_probs = routing_probs / top_k_probs.sum(-1, keepdim=True)
        return routing_probs, routing_map


class Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Parameter(
            torch.zeros(
                self.config.num_experts,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
            )
        )
        self.up_proj = nn.Parameter(
            torch.zeros(
                self.config.num_experts,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.zeros(
                self.config.num_experts,
                self.config.moe_intermediate_size,
                self.config.hidden_size,
            )
        )

    def forward(self, expert_tokens, expert_tokens_count):
        output = []

        for i in range(self.config.num_experts):
            chunk_start = 0 if i == 0 else expert_tokens_count[i - 1]
            chunk_end = expert_tokens_count[i]

            if chunk_start == chunk_end:
                continue

            chunk = expert_tokens[chunk_start:chunk_end]
            output_chunk = (
                F.silu(chunk @ self.gate_proj[i, ...]) * (chunk @ self.up_proj[i, ...])
            ) @ self.down_proj[i, ...]
            output.append(output_chunk)

        return torch.concat(output, 0)


class MOE(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config
        self.router = Router(config)
        self.experts = Experts(config)

    def forward(self, x):
        bsz, seq = x.shape[:2]
        routing_probs, routing_map = self.router(x)

        flat_routing_map = routing_map.view(-1, self.config.num_experts).T
        flat_routing_probs = routing_probs.view(-1, self.config.num_experts).T
        flat_x = x.view(-1, self.config.hidden_size)
        expert_token_idxs = torch.masked_select(
            torch.arange(bsz * seq).expand(self.config.num_experts, -1).to(x.device),
            flat_routing_map,
        )
        expert_token_probs = torch.masked_select(flat_routing_probs, flat_routing_map)
        expert_tokens = torch.index_select(flat_x, 0, expert_token_idxs)
        expert_tokens_count = flat_routing_map.sum(dim=1).cumsum(0)

        ffn_tokens = self.experts(expert_tokens, expert_tokens_count)
        print(
            "ffn_tokens ",
            ffn_tokens.shape,
            " flat_routing_probs ",
            flat_routing_probs.shape,
        )
        ffn_tokens = ffn_tokens * expert_token_probs.unsqueeze(-1).expand(
            -1, self.config.hidden_size
        )

        output = torch.zeros(bsz * seq, self.config.hidden_size).scatter_add_(
            0,
            expert_token_idxs.unsqueeze(-1).expand(-1, self.config.hidden_size),
            ffn_tokens,
        )
        return output
