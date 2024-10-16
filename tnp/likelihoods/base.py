from abc import ABC, abstractmethod

import torch
import torch.distributions as td
from check_shapes import check_shapes
from torch import nn


class Likelihood(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> td.Distribution:
        raise NotImplementedError


class UniformMixtureLikelihood(Likelihood):
    def __init__(
        self,
        likelihood: Likelihood,
    ):
        super().__init__()
        self.likelihood = likelihood

    @check_shapes("x: [..., s, n, d]")
    def forward(self, x: torch.Tensor) -> td.MixtureSameFamily:
        dists = self.likelihood(x)
        assert (
            dists.batch_shape[:-1] == x.shape[:-1]
        ), "Only works for fully independent likelihoods."

        # Re-interpret last two batch-dimensions as event dims.
        comp_dists = td.Independent(dists, 2)

        # Uniform mixture of distributions.
        mix_dists = td.Categorical(torch.ones(x.shape[:-2]).to(x))

        return td.MixtureSameFamily(mix_dists, comp_dists)
