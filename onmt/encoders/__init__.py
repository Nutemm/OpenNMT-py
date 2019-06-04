"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.local_transformer import LocalTransformerEncoder
from onmt.encoders.transformer_with_convs import TransformerWithConvsEncoder
from onmt.encoders.transformer_1st_elt import Transformer1stEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder, 'local_transformer':LocalTransformerEncoder, 
           'transformer_with_convs':TransformerWithConvsEncoder, "transformer_1st_elt": Transformer1stEncoder
           }

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc"]
