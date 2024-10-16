import copy
import warnings
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from .attention_layers import MultiHeadCrossAttentionLayer, MultiHeadSelfAttentionLayer


class TLNPTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        sample_last_layer: bool = True,
        sample_first_layer: bool = False,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)

        self.sample_last_layer = sample_last_layer
        self.sample_first_layer = sample_first_layer

        self.log_var_mhsa_layer = None
        self.log_var_mhsa_layers = None
        if sample_last_layer or sample_first_layer:
            self.log_var_mhsa_layer = mhsa_layer
        else:
            self.log_var_mhsa_layers = _get_clones(mhsa_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, s, nt, d]"
    )
    def forward(
        self,
        xc: torch.Tensor,
        xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        num_batches = xc.shape[0]
        if not (self.sample_last_layer or self.sample_first_layer):
            xc = einops.repeat(xc, "m n d -> m s n d", s=num_samples)
            xt = einops.repeat(xt, "m n d -> m s n d", s=num_samples)
            xc = einops.rearrange(xc, "m s n d -> (m s) n d")
            xt = einops.rearrange(xt, "m s n d -> (m s) n d")

        for i, (mhca_layer, mhsa_layer) in enumerate(
            zip(self.mhca_layers, self.mhsa_layers)
        ):
            xc_loc = mhsa_layer(xc)
            if not (self.sample_last_layer or self.sample_first_layer):
                assert self.log_var_mhsa_layers is not None
                xc_log_var = self.log_var_mhsa_layers[i](xc)
                xc_dist = get_dist(torch.cat([xc_loc, xc_log_var], dim=-1))
                xc = xc_dist.rsample()
            elif self.sample_last_layer and i == len(self.mhca_layers) - 1:
                assert self.log_var_mhsa_layer is not None
                xc_log_var = self.log_var_mhsa_layer(xc)
                xc_dist = get_dist(torch.cat([xc_loc, xc_log_var], dim=-1))
                xc = xc_dist.rsample(sample_shape=torch.Size((num_samples,)))
                xc = einops.rearrange(xc, "s m n d -> (m s) n d")
                xt = einops.repeat(xt, "m n d -> (m s) n d", s=num_samples)
            elif self.sample_first_layer and i == 0:
                assert self.log_var_mhsa_layer is not None
                xc_log_var = self.log_var_mhsa_layer(xc)
                xc_dist = get_dist(torch.cat([xc_loc, xc_log_var], dim=-1))
                xc = xc_dist.rsample(sample_shape=torch.Size((num_samples,)))
                xc = einops.rearrange(xc, "s m n d -> (m s) n d")
                xt = einops.repeat(xt, "m n d -> (m s) n d", s=num_samples)
            else:
                xc = xc_loc

            xt = mhca_layer(xt, xc)

        xt = einops.rearrange(xt, "(m s) n d -> m s n d", s=num_samples, m=num_batches)

        return xt


def get_dist(
    x: torch.Tensor, min_noise: float = 0.0
) -> torch.distributions.Distribution:
    loc, log_var = torch.chunk(x, 2, dim=-1)
    scale = (
        nn.functional.softplus(log_var) ** 0.5  # pylint: disable=not-callable
        + min_noise
    )
    return torch.distributions.Normal(loc, scale)


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
