import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, sin, cos):
    head_dim = x.shape[-1]
    rotated_x = torch.cat([-x[..., head_dim // 2 :], x[..., : head_dim // 2]], dim=-1)
    return cos * x + sin * rotated_x


class Rope(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.register_buffer(
            "theta",
            1
            / (
                self.rope_theta ** ((torch.arange(0, self.head_dim, 2)) / self.head_dim)
            ),
        )

    def forward(self, x):
        position_thetas = torch.einsum(
            "i,j->ij",
            torch.arange(self.max_position_embeddings),
            self.theta.repeat(2),
        )
        return torch.sin(position_thetas)[None, None, :, :], torch.cos(position_thetas)[
            None, None, :, :
        ]


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_key_value_heads, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_key_value_heads, bias=False
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(self.max_position_embeddings, self.max_position_embeddings),
                1,
            ).bool(),
        )

    def forward(self, x, sin, cos):
        bsz, seq = x.shape[:2]
        repeat_kv = self.num_attention_heads // self.num_key_value_heads

        q = self.q_norm(
            self.q_proj(x).reshape(bsz, seq, self.num_attention_heads, self.head_dim)
        ).transpose(1, 2)
        q = apply_rope(q, sin, cos)

        k = self.k_proj(x).reshape(bsz, seq, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(
            torch.repeat_interleave(
                k, repeat_kv, dim=2, output_size=self.num_attention_heads
            )
        ).transpose(1, 2)
        k = apply_rope(k, sin, cos)

        v = self.v_proj(x).reshape(bsz, seq, self.num_key_value_heads, self.head_dim)
        v = torch.repeat_interleave(
            v, repeat_kv, dim=2, output_size=self.num_attention_heads
        ).transpose(1, 2)

        o = (
            F.softmax(
                ((q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)).masked_fill(
                    self.causal_mask, -float("inf")
                ),
                dim=-1,
            )
            @ v
        ).transpose(1, 2)
        return self.o_proj(o.reshape(bsz, seq, self.hidden_size))
