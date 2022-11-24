from transformers import RobertaPreTrainedModel, RobertaModel
from torch import nn
from einops import einsum, rearrange, reduce
from rotary_embedding import RotaryEmbedding
import torch


class RobertGlobalPointer(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertGlobalPointer, self).__init__(config)
        self.roberta = RobertaModel(config)

        self.config = config
        self.head_size = config.head_size
        self.rotary_emb = RotaryEmbedding(config.head_size)
        self.pos = nn.Linear(config.hidden_size, config.head * config.head_size * 2)
        self.start_pos = nn.Linear(config.hidden_size, config.head * config.head_size)
        self.end_pos = nn.Linear(config.hidden_size, config.head * config.head_size)

    def forward(self, x, mask=None, token_type_ids=None):
        seqlen, device = x.shape[-1], x.device
        # batch, seq, dim
        x = self.roberta(x, attention_mask=mask).last_hidden_state

        # start, end = torch.split(self.pos(x), self.config.head*self.config.head_size, dim=-1)
        start = rearrange(self.start_pos(x), 'b m (c h) -> b m c h', h=self.head_size)
        end = rearrange(self.end_pos(x), 'b m (c h) -> b m c h', h=self.head_size)
        start, end = map(self.rotary_emb, (start, end))

        # batch, seq, seq
        x = einsum(start, end, 'b m h d, b n h d -> b h m n')

        # pad sequence
        mask = rearrange(mask, 'b s -> b 1 1 s').bool()
        x = x.masked_fill(~mask, -1e12)
        mask = rearrange(mask, 'b 1 1 s -> b 1 s 1').bool()
        x = x.masked_fill(~mask, -1e12)

        # pad lower tril
        tril_mask = torch.ones((seqlen, seqlen), device=device).triu().bool()
        x = x.masked_fill(~tril_mask, -1e12)

        return x / self.head_size**0.5