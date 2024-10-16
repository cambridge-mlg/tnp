import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.attention_layers import MultiHeadAttentionLayer
from ..networks.transformer import TransformerEncoder
from .base import ConditionalNeuralProcess
from .tnp import TNPDecoder


class ANPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TransformerEncoder,
        mha_layer: MultiHeadAttentionLayer,
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.mha_layer = mha_layer
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        yc_encoded = self.y_encoder(yc)

        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zc = self.xy_encoder(zc)

        # Self-attention layers on context tokens.
        zc = self.transformer_encoder(zc)

        # Cross-attention layer with input locations as query/keys.
        zt = self.mha_layer(xt_encoded, xc_encoded, zc)

        return zt


class ANP(ConditionalNeuralProcess):
    def __init__(self, encoder: ANPEncoder, decoder: TNPDecoder, likelihood: nn.Module):
        super().__init__(encoder, decoder, likelihood)
