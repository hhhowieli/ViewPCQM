import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, query=None, **kwargs):
        if query != None:
            return self.fn(self.norm1(x), self.norm2(query), **kwargs)
        return self.fn(self.norm1(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MHAttention(nn.Module):
    def __init__(self, input_dim, head_dim, num_heads, drop_rate=0.1) -> None:
        super().__init__()

        project_out = not (num_heads == 1 and head_dim == input_dim)

        self.num_heads = num_heads
        self.scale = head_dim**(-0.5)

        inner_dim = head_dim * num_heads

        self.to_kv = nn.Linear(input_dim, inner_dim * 2, bias = False)
        self.to_q = nn.Linear(input_dim, inner_dim, bias = False)

        self.dropout = nn.Dropout(drop_rate)
        self.softmax = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, input_dim),
            nn.Dropout(drop_rate)
        ) if project_out else nn.Identity()

    def forward(self, x, query=None):


        q = self.to_q(query if query != None else x)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.num_heads)
        kv = self.to_kv(x).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), kv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)

        return out
    

class TransformerEncoder(nn.Module):
    def __init__(self, depth, input_dim, num_heads, head_dim, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(input_dim, MHAttention(input_dim, num_heads=num_heads, head_dim=head_dim, drop_rate=dropout)),
                PreNorm(input_dim, FeedForward(input_dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        resual = x

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return resual + x


class TransformerDecoder(nn.Module):
    def __init__(self, depth, input_dim, num_heads, head_dim, mlp_dim, dropout = 0.) -> None:
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(input_dim, MHAttention(input_dim, num_heads=num_heads, head_dim=head_dim, drop_rate=dropout)),
                PreNorm(input_dim, FeedForward(input_dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, kv, query):
        x = query

        for attn, ff in self.layers:
            x = attn(kv, x) + x
            x = ff(x) + x

        return x + query




if __name__=="__main__":

    for _ in range(10):
        x = torch.rand((4, 512*6, 64)).cuda()

        q = torch.rand((4, 512, 64)).cuda()

        encoder = TransformerEncoder(depth=2, input_dim=64, num_heads=6, head_dim=384, mlp_dim=512).cuda()

        decoder = TransformerDecoder(depth=4, input_dim=64, num_heads=8, head_dim=384, mlp_dim=512).cuda()

        x = encoder(x)
        y = decoder(x, q)

        print(x.shape)
        print(y.shape)



