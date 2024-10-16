from typing import Union

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import TransformerEncoder
from .anp import ANPEncoder
from .base import LatentNeuralProcess
from .tnp import TNPDecoder, TNPEncoder


class ALNPEncoder(nn.Module):
    def __init__(
        self,
        *,
        tnp_encoder: Union[TNPEncoder, ANPEncoder],
        latent_transformer_encoder: TransformerEncoder,
        latent_xy_encoder: nn.Module,
        latent_x_encoder: nn.Module = nn.Identity(),
        latent_y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.tnp_encoder = tnp_encoder
        self.latent_transformer_encoder = latent_transformer_encoder
        self.latent_xy_encoder = latent_xy_encoder
        self.latent_x_encoder = latent_x_encoder
        self.latent_y_encoder = latent_y_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, s, nt, dz]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        # Get deterministic encoding.
        zt = self.tnp_encoder(xc, yc, xt)

        # Get latent path encoding.
        xc_encoded = self.latent_x_encoder(xc)
        yc_encoded = self.latent_y_encoder(yc)
        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zc = self.latent_xy_encoder(zc)
        zc = self.latent_transformer_encoder(zc)

        # Get latent path distribution.
        latent_params = zc.mean(-2)
        latent_loc, latent_log_var = torch.chunk(latent_params, 2, dim=-1)
        latent_scale = (
            nn.functional.softplus(latent_log_var)  # pylint: disable=not-callable
            ** 0.5
            + 1e-3
        )
        latent_dist = torch.distributions.Normal(latent_loc, latent_scale)

        # Shape (num_samples, m, dz)
        latent_samples = latent_dist.rsample(sample_shape=torch.Size((num_samples,)))
        # Shape (num_samples, m, nt, dz)
        latent_samples = einops.repeat(
            latent_samples, "s m d -> s m nt d", nt=xt.shape[1]
        )

        # Shape (num_samples, m, nt, dz)
        zt = einops.repeat(zt, "m nt dz -> s m nt dz", s=num_samples)

        # Concatenate together.
        zt = torch.cat((zt, latent_samples), dim=-1)

        # Move sample dimension.
        zt = einops.rearrange(zt, "s m nt dz -> m s nt dz")

        return zt


class ALNP(LatentNeuralProcess):
    def __init__(
        self, encoder: ALNPEncoder, decoder: TNPDecoder, likelihood: nn.Module
    ):
        super().__init__(encoder, decoder, likelihood)
