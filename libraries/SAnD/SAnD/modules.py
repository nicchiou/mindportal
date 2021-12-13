import math

import torch
import numpy as np
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """ Applies positional encoding to the input tensor using addition. """
    def __init__(self, d_model, seq_len) -> None:
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(
                    pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i+1] = math.cos(
                    pos / (10000 ** ((2 * (i+1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x) -> torch.Tensor:
        seq_len = x.shape[1]
        # Mutliply output of embedding weights by sqrt(d_model)
        x = math.sqrt(self.d_model) * x
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        return x


class ResidualBlock(nn.Module):
    """
    Applies a residual connection to the output of the layer specified during
    initialization by adding the input to the layer to the output of the
    layer and performing layer normalization.
    """
    def __init__(self, layer: nn.Module, d_model: int, p=0.1) -> None:
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(d_model)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, seq_len, features]
        :return: [N, seq_len, features]
        """
        if isinstance(self.layer, nn.MultiheadAttention):
            src = x.transpose(0, 1)             # [seq_len, N, features]
            output, self.attn_weights = self.layer(src, src, src)
            output = output.transpose(0, 1)     # [N, seq_len, features]

        else:
            output = self.layer(x)

        output = self.dropout(output)
        output = self.norm(x + output)  # Add and norm
        return output


class PositionWiseFeedForward(nn.Module):
    """
    Module expands the dimmension to a twice the hidden size and back down to
    the original hidden size.
    """
    def __init__(self, hidden_size: int) -> None:
        super(PositionWiseFeedForward, self).__init__()

        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.transpose(1, 2)
        tensor = self.conv(tensor)
        tensor = tensor.transpose(1, 2)

        return tensor


class EncoderBlock(nn.Module):
    """
    The encoder block consists of a multi-head attention module with a residual
    connection (add & norm) and a point-wise feed-forward module with a
    residual connection (add & norm).

    `Attention is All You Need <http://arxiv.org/abs/1711.03905>`
    Vaswani et al.
    """
    def __init__(
            self, embed_dim: int, num_head: int, dropout_rate=0.1) -> None:
        """
        :param embed_dim: dimension of the input embedding
        :param num_head: number of attention heads
        :param dropout_rate: dropout applied before add & norm in residual
        """
        super(EncoderBlock, self).__init__()

        self.attention = ResidualBlock(
            nn.MultiheadAttention(embed_dim, num_head), embed_dim,
            p=dropout_rate
        )
        self.ffn = ResidualBlock(
            PositionWiseFeedForward(embed_dim), embed_dim, p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.ffn(x)
        return x


class DenseInterpolation(nn.Module):
    """
    Summarization of embeddings over all time steps for the sequence while
    preserving the order and reducing the dimensionality (avoiding naive
    concatenation of embeddings). Determines weights of each time step's
    embedding s_t to the output vector with factor M (length seq_len --> M).
    """
    def __init__(self, seq_len: int, factor: int) -> None:
        """
        :param seq_len: sequence length
        :param factor: factor M
        """
        super(DenseInterpolation, self).__init__()

        W = np.zeros((factor, seq_len), dtype=np.float32)

        # Algorithm from Attend and Diagnose 2017
        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1 + m)) / factor),
                               dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = torch.tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.W.repeat(x.shape[0], 1, 1).requires_grad_(False)
        u = torch.bmm(w, x)
        return u.transpose_(1, 2)


class ClassificationModule(nn.Module):
    """
    Takes the output of dense interpolation and pases it through a fully-
    connected classification layer.
    """
    def __init__(self, d_model: int, factor: int) -> None:
        """
        :param d_model: dimension of the model embeddings
        :param factor: factor for output of dense interpolation (M)
        :param num_classes: number of classes in the classification task
        """
        super(ClassificationModule, self).__init__()

        self.d_model = d_model
        self.factor = factor

        self.fc = nn.Linear(int(d_model * factor), 1)

        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(-1, int(self.factor * self.d_model))
        x = self.fc(x)
        return x
