import torch
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torch import nn
from einops import einsum, rearrange, reduce
from rotary_embedding import RotaryEmbedding


class BertGlobalPointer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertGlobalPointer, self).__init__(config)
        self.bert = BertModel(config)

        self.config = config
        self.head_size = config.head_size
        self.rotary_emb = RotaryEmbedding(config.head_size)
        self.pos = nn.Linear(config.hidden_size, config.head * config.head_size * 2)
        self.start_pos = nn.Linear(config.hidden_size, config.head * config.head_size)
        self.end_pos = nn.Linear(config.hidden_size, config.head * config.head_size)

    def forward(self, x, mask=None, token_type_ids=None):
        seqlen, device = x.shape[-1], x.device
        # batch, seq, dim
        x = self.bert(x, token_type_ids=token_type_ids, attention_mask=mask).last_hidden_state

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


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE
        self.rotary_emb = RotaryEmbedding(inner_dim)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # TODO:修改为Linear获取？

        if self.RoPE:
            # # pos_emb:(batch_size, seq_len, inner_dim)
            # pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            # cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            # sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1)
            # qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
            # qw2 = qw2.reshape(qw.shape)
            # qw = qw * cos_pos + qw2 * sin_pos
            # kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
            # kw2 = kw2.reshape(kw.shape)
            # kw = kw * cos_pos + kw2 * sin_pos
            qw, kw = map(self.rotary_emb, (qw, kw))

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

