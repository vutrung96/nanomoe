import torch.nn as nn

from nanomoe.layers.block import Block


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tied_embedding:
            self.embedding.weight = self.lm_head.weight
        self.rms_f = nn.RMSNorm(config.hidden_size)

    def forward(self, x, mask):
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.lm_head(self.rms_f(x))
        return logits
