"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch.nn.functional as F
import math

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward


class LocalTransformerEncoderLayer(nn.Module):
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
        super(LocalTransformerEncoderLayer, self).__init__()

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


class LocalTransformerEncoder(EncoderBase):
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
        super(LocalTransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [LocalTransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.kernel_size = 10
        self.with_shifts = True

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

        #initialization
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous() #[B, T, C]
        words = src[:, :, 0].transpose(0, 1).contiguous()
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx

        #Shorten sentences w.r.t. kernel size 
        batch_size, src_len, d = out.size()
        size_to_add = (self.kernel_size - src_len % self.kernel_size) % self.kernel_size

        out = F.pad(out,(0,0,size_to_add,0,0,0)) #pad on the left of 2nd dimension (i.e. T) 
        words_data = F.pad(words.data,(size_to_add,0), value=padding_idx) #pad on the left of last dimension (i.e. T) 

        mask = words_data.eq(padding_idx).unsqueeze(1) #put 1's on the tokens that shouldn't be considered 

        if self.with_shifts:
            mask_with_shifts = F.pad(mask, (math.ceil(self.kernel_size / 2), math.floor(self.kernel_size / 2)), value=1)

            #Test about the masks
            print("len sentence: {}\nNormal mask: {}\nMask with shifts: {}\n".format(src_len,mask[0],mask_with_shifts[0]))   
            mask_with_shifts = mask_with_shifts.view(-1, 1, self.kernel_size)

        #resize so that T becomes kernel_size
        out = out.view(-1, self.kernel_size, d)
        mask = mask.view(-1, 1, self.kernel_size)


        for n_layer, layer in enumerate(self.transformer):
            
            if self.with_shifts and n_layer % 2 ==1:
                out = out.view(batch_size, -1, d) #resize into the initial dimension
                # print(out[0], "\n", out.size()) 
                out = F.pad(out, (0, 0, math.ceil(self.kernel_size / 2), math.floor(self.kernel_size / 2), 0, 0)) #add more padding to shift
                # print(out[0][5:-5], "\n", out.size())
                out = out.view(-1, self.kernel_size, d)

                out = layer(out, mask_with_shifts)
                
                out = out.view(batch_size, -1, d) #resize into the initial dimension
                out = out[:, math.ceil(self.kernel_size / 2):-math.floor(self.kernel_size / 2), :] #delete added padding to shift
                out = out.view(-1, self.kernel_size, d)

            else:
                out = layer(out, mask)


        out = out.view(batch_size, src_len+size_to_add, -1) #resize into the initial dimension
        out = out[:,size_to_add:,:] #remove initial padding
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths
