from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


class SinCosPositionEncoding(nn.Module):

    def __init__(self, context_size, embed_size) -> None:
        super().__init__()
        # generate the positional encoding to select from
        self.encoding = nn.Embedding.from_pretrained(
            self.get_sin_cos_embeddings(
                context_size=context_size, embed_size=embed_size
            )
        )

    @staticmethod
    def get_sin_cos_embeddings(context_size, embed_size) -> torch.Tensor:
        emb = torch.zeros(context_size, embed_size, requires_grad=False)
        pos = torch.arange(context_size, requires_grad=False)
        for i in range(embed_size // 2):
            emb[:, 2 * i] = torch.sin(pos / 10000 ** (2 * i / embed_size))
            emb[:, 2 * i + 1] = torch.cos(pos / 10000 ** (2 * i / embed_size))
        return emb

    def forward(self, t):  # t is Tensor dims (T,)
        return self.encoding(t)


class AttentionHead(nn.Module):
    # B: batch size
    # T: time dimension - context_len
    # E: embed_size
    # H: head dimension - head_size
    # N: num heads

    def __init__(
        self,
        context_size: int,
        embed_size: int,
        head_size: int,
        dropout: Optional[float] = None,
        decoder: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size  # H
        self.decoder = decoder
        self.query = nn.Linear(embed_size, head_size, bias=False)  # (E, H)
        self.key = nn.Linear(embed_size, head_size, bias=False)  # (E, H)
        self.value = nn.Linear(embed_size, head_size, bias=False)  # (E, H)
        self.register_buffer(
            "mask", torch.tril(torch.ones(context_size, context_size))
        )  # (T, T)
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x):
        _, T, _ = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)

        weights = (
            q @ k.transpose(-2, -1) * self.head_size**-0.5
        )  # (B, T, H) @ (B, H, T) -> (B, T, T)
        if self.decoder:
            # NOTE: T x T section of mask for flexibility in input dims.
            self.mask: torch.Tensor
            weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        # over the context_len dimension -> (B, context_len, context_len) with each
        # row summing to 1
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        # dropout over the weights (regularization)
        # todo maybe experiment with this more.
        if self.dropout:
            weights = self.dropout(weights)
        out = weights @ v  # (B, T, H)
        return out


class MultiHeadAttention(nn.Module):
    # B: batch size
    # T: time dimension - context_len
    # E: embed_size
    # H: head dimension - head_size
    # N: num heads

    def __init__(
        self,
        context_size: int,
        embed_size: int,
        head_size: int,
        n_heads: int,
        dropout: Optional[float] = None,
        decoder: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size  # H
        self.heads = nn.ModuleList(
            AttentionHead(
                context_size=context_size,
                embed_size=embed_size,
                head_size=head_size,
                dropout=dropout,
                decoder=decoder,
            )
            for _ in range(n_heads)
        )
        # this layer ensures that the output dim is always head_size.  Is that what I
        # want? maybe should be embed_size
        # seems like one rationale is that this projects back into the residual
        # pathway. To do with gradients.
        self.out_layer = nn.Linear(
            in_features=head_size * n_heads, out_features=embed_size
        )
        # dropout over the projection -
        # todo experiment with this. Think about interpretability?
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, H * N)
        x = self.out_layer(x)  # (B, T, E)
        if self.dropout:
            x = self.dropout(x)
        return x


class FeedforwardNet(nn.Module):
    def __init__(
        self, embed_size: int, hidden_size: int, dropout: Optional[float] = None
    ):
        super().__init__()
        ff_args = [
            nn.Linear(in_features=embed_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=embed_size),
        ]
        # add dropout if configured
        if dropout:
            ff_args.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*ff_args)

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        context_size: int,
        embed_size: int,
        head_size: int,
        hidden_size: int,
        n_heads: int,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.attention_head = MultiHeadAttention(
            context_size=context_size,
            embed_size=embed_size,
            head_size=head_size,
            n_heads=n_heads,
            dropout=dropout,
            decoder=True,
        )
        self.attention_norm = nn.LayerNorm(embed_size)
        self.feedforward = FeedforwardNet(
            embed_size=embed_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.feedforward_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        # this time we will add the residual connections and norm layers
        # x is (B, T)
        x = x + self.attention_head(self.attention_norm(x))
        out = x + self.feedforward(self.feedforward_norm(x))
        return out


class ChunkStackLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, chunk_size: int) -> None:
        super().__init__()
        self.lins = nn.ModuleList(
            [
                nn.Linear(in_features=in_features, out_features=out_features)
                for _ in range(chunk_size)
            ]
        )

    def forward(self, x):
        # one option is n Linear and stack the outputs
        n_dims = len(x.shape)
        output = torch.stack([lin(x) for lin in self.lins], dim=n_dims - 1)
        return output


class ChunkCatLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, chunk_size: int) -> None:
        super().__init__()
        self.lin = nn.Linear(
            in_features=in_features, out_features=out_features * chunk_size
        )

    def forward(self, x):
        # other option is Linear with n_chunks * out_features as the output dim.
        output = self.lin(x)
        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn(4, 8, 2)
    head = AttentionHead(context_size=8, embed_size=2, head_size=4, decoder=True)
    print(head(x))

    x_multi = torch.randn(4, 8, 2)
    multi_head = MultiHeadAttention(
        context_size=8, embed_size=2, head_size=4, n_heads=4, decoder=True
    )
    print(multi_head(x_multi))

    x = torch.randn(2, 4, 6)  # (B, T, E)
    n_chunks = 3

    stacked = ChunkStackLinear(in_features=6, out_features=7, chunk_size=n_chunks)

    catted = ChunkCatLinear(in_features=6, out_features=7, chunk_size=n_chunks)

    s = stacked(x)
    print(s)
    c = catted(x)
    print(c)
