import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.latent_transformer import TLNPTransformerEncoder
from ..utils.helpers import preprocess_observations
from .base import LatentNeuralProcess
from .tnp import TNPDecoder


class TLNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TLNPTransformerEncoder,
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, s, nt, dz]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)

        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zt = torch.cat((xt_encoded, yt_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        zt = self.transformer_encoder(zc, zt, num_samples=num_samples)
        return zt


class TLNP(LatentNeuralProcess):
    def __init__(
        self, encoder: TLNPEncoder, decoder: TNPDecoder, likelihood: nn.Module
    ):
        super().__init__(encoder, decoder, likelihood)
