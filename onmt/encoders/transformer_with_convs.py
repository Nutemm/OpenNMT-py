"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerWithConvsEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions):
        super(TransformerWithConvsEncoder, self).__init__()
        
        self.embeddings = embeddings

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])

        self.first_conv = nn.Conv1d(d_model, d_model, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.nb_conv_blocks = 6
        self.conv_layers = [nn.Conv1d(d_model, d_model, 3, stride=1, padding=1, dilation=1, groups=1, bias=True) 
                        for _ in range(self.nb_conv_blocks)]
        self.batch_norm_layers = [nn.BatchNorm1d(d_model) for _ in range(self.nb_conv_blocks)]

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.first_time = True #to transform the model into a cuda one if necessary

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx

        if self.first_time:
            self.first_time = False
            if out.is_cuda:
                for i in range(len(self.conv_layers)):
                    self.conv_layers[i] = self.conv_layers[i].cuda()
                    self.batch_norm_layers[i] = self.batch_norm_layers[i].cuda()
                self.first_conv = self.first_conv.cuda()

        # print(out.size())
        # print(out)
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        #Apply convolutions first
        out = out.transpose(1,2).contiguous()
        out = self.first_conv(out)
        for i in range(self.nb_conv_blocks):
            out = F.relu(self.batch_norm_layers[i](self.conv_layers[i](out)))
        out = out.transpose(1,2).contiguous()

        # print(out.size())

        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths
