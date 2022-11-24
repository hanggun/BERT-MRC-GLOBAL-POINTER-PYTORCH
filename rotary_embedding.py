from torch import nn
import torch
from torch import einsum
from einops import repeat, rearrange


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        embeddings = torch.stack([freqs.sin(), freqs.cos()], dim=-1)
        return embeddings.reshape(max_seq_len, self.dim)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.sin_emb = SinusoidalPositionEmbedding(dim)
        self.register_buffer('pos_emb', None, persistent=False)

    def get_pos_emb(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.sin_emb(n, device=device)
        self.register_buffer('pos_emb', pos_emb, persistent=False)
        return pos_emb

    def forward(self, t):
        n, device = t.shape[1], t.device

        pos_emb = self.get_pos_emb(n, device)
        if t.ndim == 4:
            pos_emb = rearrange(pos_emb, 'm n -> m 1 n')
            cos_pos = pos_emb[..., 1::2, None].expand(-1, -1, -1, 2).flatten(-2, -1)
            sin_pos = pos_emb[..., ::2, None].expand(-1, -1, -1, 2).flatten(-2, -1)
            t2 = torch.stack([-t[..., 1::2], t[..., ::2]], dim=-1)
            t2 = rearrange(t2, 'b m n h c -> b m n (h c)')
        else:
            cos_pos = pos_emb[..., 1::2, None].expand(-1, -1, 2).flatten(-2, -1)
            sin_pos = pos_emb[..., ::2, None].expand(-1, -1, 2).flatten(-2, -1)
            t2 = torch.stack([-t[..., 1::2], t[..., ::2]], dim=-1)
            t2 = rearrange(t2, 'b m h c -> b m (h c)')
        # cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1) # 速度慢，改用expand+flatten
        # sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

        t = t * cos_pos + t2 * sin_pos

        return t
