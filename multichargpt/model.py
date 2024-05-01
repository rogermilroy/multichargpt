from typing import Any
import torch

from abc import ABC, abstractmethod
from torch import Tensor, nn
from torch.nn import functional as F

from components import (
    ChunkCatLinear,
    ChunkStackLinear,
    TransformerBlock,
    SinCosPositionEncoding,
)


class LanguageModel(ABC):

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor: ...

    @abstractmethod
    def loss(self, logits: Any, targets: Any) -> Tensor: ...

    @abstractmethod
    def generate(self, *args, **kwargs) -> Tensor: ...


class TorchLanguageModel(nn.Module, LanguageModel): ...


class BigramLanguageModel(TorchLanguageModel):
    def __init__(self, vocab_size):
        super().__init__()
        # vocab_size x vocab_size table (direct probs for each char based on
        # previous char) like Q table.
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )

    def forward(self, x):
        # x is (B, T)
        return self.token_embedding(x)  # (B, T, C)

    def loss(self, logits, targets):
        b, t, c = logits.shape
        logits = logits.view((b * t, c))
        targets = targets.view(b * t)
        return F.cross_entropy(logits, targets)

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(x)  # logits (B, T, C) C is output options

            logits = logits[:, -1, :]  # select last time step from logits (B, 1, C)
            probs = F.softmax(logits, dim=-1)  # logits to probs
            x_next = torch.multinomial(
                probs, num_samples=1
            )  # select one from probs (B, 1, 1)

            x = torch.cat((x, x_next), dim=1)  # (B, T + 1, 1)
        return x


class TransformerMultiBlockLanguageModel(TorchLanguageModel):
    # B: batch size
    # T: time dimension - context_len
    # E: embed_size
    # H: head dimension - head_size
    # N: num heads
    # Ch: chunk_size - number of tokens per chunk

    def __init__(
        self,
        context_size,
        vocab_size,
        embed_size,
        head_size,
        hidden_size,
        n_heads,
        n_blocks,
        dropout,
        pos_embedding="sin_cos",
    ):
        """
        Creates a Multi Block Transformer model. That is a stack of MultiHeadAttention
        Blocks parameterised accordingly.
        :param context_size: dimension of the context (ie. length of an input string)
        :param vocab_size: dimension of the vocabulary
        :param embed_size: dimension of the token and position embeddings
        :param head_size: dimension of the attention head - usually computed from
            embed_size and n_heads - embed_size // n_heads.
            Keeping separate for experimentation.
        :param hidden_size: dimension of the feedforward networks in the Attention
            blocks
        :param n_heads: number of attention heads per block
        :param n_blocks: number of blocks - analogous to depth of the Transformer
        :param dropout: proportion of dropout applied (to Attention heads and
            feedforward nets)
        :param pos_embedding: the position embedding technique used
            {sin_cos | learned} default sin_cos.
        """
        super().__init__()
        self.context_size = context_size
        self.head_size = head_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        if pos_embedding == "learned":
            self.position_embedding = nn.Embedding(context_size, embed_size)
        elif pos_embedding == "sin_cos":
            self.position_embedding = SinCosPositionEncoding(
                context_size=context_size, embed_size=embed_size
            )
        else:
            raise ValueError(
                f"pos_embedding must be one of 'learned' or 'sin_cos' "
                f"found: {pos_embedding}"
            )
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    context_size=context_size,
                    embed_size=embed_size,
                    head_size=head_size,
                    n_heads=n_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
            ]
            * n_blocks
        )
        self.layer_norm = nn.LayerNorm(embed_size)
        # TODO tie these weights to token embedding
        self.output_layer = nn.Linear(in_features=embed_size, out_features=vocab_size)

    def forward(self, x):
        # this time we will add the residual connections and norm layers
        # x is (B, T)
        _, t = x.shape
        x = self.token_embedding(x)  # (B, T, E)
        pos = self.position_embedding(torch.arange(t, device=x.device))  # (T, E)
        x = x + pos
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        out = self.output_layer(x)
        return out

    def loss(self, logits, targets):
        b, t, c = logits.shape
        logits = logits.view((b * t, c))
        targets = targets.view(b * t)
        return F.cross_entropy(logits, targets)

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # left trim x to be last n_context tokens
            x_trim = x[:, -self.context_size :]

            logits = self(x_trim)  # logits (B, T, E) E is output options

            logits = logits[:, -1, :]  # select last time step from logits (B, 1, E)
            probs = F.softmax(logits, dim=-1)  # logits to probs
            x_next = torch.multinomial(
                probs, num_samples=1
            )  # select one from probs (B, 1, 1)

            x = torch.cat((x, x_next), dim=1)  # (B, T + 1, 1)
        return x


class TransformerFixedLookahead(TorchLanguageModel):
    # B: batch size
    # T: time dimension - context_len
    # E: embed_size
    # H: head dimension - head_size
    # N: num heads
    # Ch: chunk_size - number of tokens per chunk

    def __init__(
        self,
        context_size: int,
        vocab_size: int,
        embed_size: int,
        head_size: int,
        hidden_size: int,
        n_heads: int,
        n_blocks: int,
        chunk_size: int,
        dropout: float,
        chunk_method: str = "cat",
        pos_embedding: str = "sin_cos",
    ):
        """
        Creates a Multi Block Transformer model. That is a stack of MultiHeadAttention
        Blocks parameterised accordingly.
        :param context_size: dimension of the context (ie. length of an input string)
        :param vocab_size: dimension of the vocabulary
        :param embed_size: dimension of the token and position embeddings
        :param head_size: dimension of the attention head - usually computed from
            embed_size and n_heads - embed_size // n_heads.
            Keeping separate for experimentation.
        :param hidden_size: dimension of the feedforward networks in the Attention
            blocks
        :param n_heads: number of attention heads per block
        :param n_blocks: number of blocks - analogous to depth of the Transformer
        :param chunk_size: the size of output chunk (input output offset)
        :param dropout: proportion of dropout applied (to Attention heads and
            feedforward nets)
        :param pos_embedding: the position embedding technique used
            {sin_cos | learned} default sin_cos.
        """
        super().__init__()
        self.context_size = context_size
        self.head_size = head_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        if pos_embedding == "learned":
            self.position_embedding = nn.Embedding(context_size, embed_size)
        elif pos_embedding == "sin_cos":
            self.position_embedding = SinCosPositionEncoding(
                context_size=context_size, embed_size=embed_size
            )
        else:
            raise ValueError(
                f"pos_embedding must be one of 'learned' or 'sin_cos' "
                f"found: {pos_embedding}"
            )
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    context_size=context_size,
                    embed_size=embed_size,
                    head_size=head_size,
                    n_heads=n_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
            ]
            * n_blocks
        )
        self.layer_norm = nn.LayerNorm(embed_size)
        # TODO tie these weights to token embedding
        self.chunk_method = chunk_method
        if self.chunk_method == "cat":
            self.output_layer = ChunkCatLinear(
                in_features=embed_size, out_features=vocab_size, chunk_size=chunk_size
            )
        elif self.chunk_method == "stack":
            self.output_layer = ChunkStackLinear(
                in_features=embed_size, out_features=vocab_size, chunk_size=chunk_size
            )
        else:
            raise ValueError("chunk_method must be one of [cat, stack]")
        self.chunk_size = chunk_size

    def forward(self, x):
        # this time we will add the residual connections and norm layers
        # x is (B, T)
        _, t = x.shape
        x = self.token_embedding(x)  # (B, T, E)
        pos = self.position_embedding(torch.arange(t, device=x.device))  # (T, E)
        x = x + pos  # (B, T, E)
        x = self.transformer_blocks(x)  # (B, T, E)
        x = self.layer_norm(x)  # (B, T, E)
        out = self.output_layer(x)  # (B, T, Ch, E)
        return out

    def _stacked_loss(self, logits, targets):
        b, t, ch, c = logits.shape
        logits = logits.view((b * t * ch, c))  # logits will be (B, T, Ch, E)
        targets = targets.view(b * t * ch)  # targets will be (B, T, Ch) initially
        return F.cross_entropy(logits, targets)

    def _catted_loss(self, logits, targets):
        b, t, c_by_ch = logits.shape
        logits = logits.view(
            (b * t * self.chunk_size, c_by_ch // self.chunk_size)
        )  # (B, T, Ch * E)
        targets = targets.view(
            b * t * self.chunk_size
        )  # targets will be (B, T, Ch) initially
        return F.cross_entropy(logits, targets)

    def loss(self, logits, targets):
        # TODO refactor to just process logits and targets - after testing this works correctly.
        if self.chunk_method == "cat":
            return self._catted_loss(logits=logits, targets=targets)
        elif self.chunk_method == "stack":
            return self._stacked_loss(logits=logits, targets=targets)
        # return F.cross_entropy(logits, targets)  # TODO restore when refactored.

    def generate(self, x, max_new_chunks):
        for _ in range(max_new_chunks):
            # left trim x to be last n_context tokens
            x_trim = x[:, -self.context_size :]

            logits = self(x_trim)  # logits (B, T, E) E is output dim

            logits = logits[
                :, -self.chunk_size :, :
            ]  # select last time step from logits (B, Ch, E)
            probs = F.softmax(
                logits, dim=-1
            )  # logits to probs - TODO check dims of this
            x_next = torch.multinomial(
                probs.squeeze(), num_samples=1  # TODO change this to chunk_size?
            )  # select one from probs (B, Ch, 1) TODO check it selects one per time step as needed

            x = torch.cat((x, x_next.T), dim=1)  # (B, T + Ch, 1)
        return x
