import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomoe.config import MoeConfig


def apply_rope(x, sin, cos):
    head_dim = x.shape[-1]
    rotated_x = torch.cat([-x[..., head_dim // 2 :], x[..., : head_dim // 2]], dim=-1)
    return cos * x + sin * rotated_x


class Rope(nn.Module):
    def __init__(self, config: MoeConfig):
        super().__init__()
        self.config = config
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.register_buffer(
            "theta",
            1 / (self.config.rope_theta ** ((torch.arange(0, head_dim, 2)) / head_dim)),
        )

    def forward(self, x):
        seq = x.shape[1]
        position_thetas = torch.einsum(
            "i,j->ij",
            torch.arange(seq),
            self.theta.repeat(2),
        )
        return torch.sin(position_thetas)[None, None, :, :], torch.cos(position_thetas)[
            None, None, :, :
        ]


class Attention(nn.Module):
    def __init__(self, config: MoeConfig):
        super().__init__()
        self.config = config
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

        self.q_proj = nn.Linear(
            self.config.hidden_size, self.config.hidden_size, bias=False
        )
        self.k_proj = nn.Linear(
            self.config.hidden_size,
            head_dim * self.config.num_key_value_heads,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.config.hidden_size,
            head_dim * self.config.num_key_value_heads,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.config.hidden_size, self.config.hidden_size, bias=False
        )

    def forward(self, x, sin, cos, mask):
        bsz, seq = x.shape[:2]
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        repeat_kv = self.config.num_attention_heads // self.config.num_key_value_heads

        q = self.q_norm(
            self.q_proj(x).reshape(bsz, seq, self.config.num_attention_heads, head_dim)
        ).transpose(1, 2)
        q = apply_rope(q, sin, cos)

        k = self.k_proj(x).reshape(bsz, seq, self.config.num_key_value_heads, head_dim)
        k = self.k_norm(
            torch.repeat_interleave(
                k, repeat_kv, dim=2, output_size=self.config.num_attention_heads
            )
        ).transpose(1, 2)
        k = apply_rope(k, sin, cos)

        v = self.v_proj(x).reshape(bsz, seq, self.config.num_key_value_heads, head_dim)
        v = torch.repeat_interleave(
            v, repeat_kv, dim=2, output_size=self.config.num_attention_heads
        ).transpose(1, 2)

        o = (
            F.softmax(
                ((q @ k.transpose(-1, -2)) / math.sqrt(head_dim)).masked_fill(
                    mask, -float("inf")
                ),
                dim=-1,
            )
            @ v
        ).transpose(1, 2)
        return self.o_proj(o.reshape(bsz, seq, self.config.hidden_size))
