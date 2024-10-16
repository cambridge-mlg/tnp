from abc import ABC

import gpytorch
import torch


class RandomHyperparameterKernel(ABC, gpytorch.kernels.Kernel):
    def sample_hyperparameters(self):
        pass


class ScaleKernel(gpytorch.kernels.ScaleKernel, RandomHyperparameterKernel):
    def __init__(
        self, min_log10_outputscale: float, max_log10_outputscale: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log10_outputscale = min_log10_outputscale
        self.max_log10_outputscale = max_log10_outputscale

    def sample_hyperparameters(self):
        # Sample outputscale.
        log10_outputscale = (
            torch.rand(()) * (self.max_log10_outputscale - self.min_log10_outputscale)
            + self.min_log10_outputscale
        )

        outputscale = 10.0**log10_outputscale
        self.outputscale = outputscale

        # Sample base kernel hyperparameters.
        self.base_kernel.sample_hyperparameters()


class RBFKernel(gpytorch.kernels.RBFKernel, RandomHyperparameterKernel):
    def __init__(
        self, min_log10_lengthscale: float, max_log10_lengthscale: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log10_lengthscale = min_log10_lengthscale
        self.max_log10_lengthscale = max_log10_lengthscale

    def sample_hyperparameters(self):
        # Sample lengthscale.
        shape = self.ard_num_dims if self.ard_num_dims is not None else ()
        log10_lengthscale = (
            torch.rand(shape)
            * (self.max_log10_lengthscale - self.min_log10_lengthscale)
            + self.min_log10_lengthscale
        )

        lengthscale = 10.0**log10_lengthscale
        self.lengthscale = lengthscale


class MaternKernel(gpytorch.kernels.MaternKernel, RandomHyperparameterKernel):
    def __init__(
        self, min_log10_lengthscale: float, max_log10_lengthscale: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log10_lengthscale = min_log10_lengthscale
        self.max_log10_lengthscale = max_log10_lengthscale

    def sample_hyperparameters(self):
        # Sample lengthscale.
        shape = self.ard_num_dims if self.ard_num_dims is not None else ()
        log10_lengthscale = (
            torch.rand(shape)
            * (self.max_log10_lengthscale - self.min_log10_lengthscale)
            + self.min_log10_lengthscale
        )

        lengthscale = 10.0**log10_lengthscale
        self.lengthscale = lengthscale


class PeriodicKernel(gpytorch.kernels.PeriodicKernel, RandomHyperparameterKernel):
    def __init__(
        self,
        min_log10_lengthscale: float,
        max_log10_lengthscale: float,
        min_log10_period: float,
        max_log10_period: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_log10_lengthscale = min_log10_lengthscale
        self.max_log10_lengthscale = max_log10_lengthscale
        self.min_log10_period = min_log10_period
        self.max_log10_period = max_log10_period

    def sample_hyperparameters(self):
        # Sample lengthscale.
        shape = self.ard_num_dims if self.ard_num_dims is not None else ()
        log10_lengthscale = (
            torch.rand(shape)
            * (self.max_log10_lengthscale - self.min_log10_lengthscale)
            + self.min_log10_lengthscale
        )

        lengthscale = 10.0**log10_lengthscale
        self.lengthscale = lengthscale

        # Sample period.
        log10_period = (
            torch.rand(shape) * (self.max_log10_period - self.min_log10_period)
            + self.min_log10_period
        )

        period = 10.0**log10_period
        self.period_length = period


class CosineKernel(gpytorch.kernels.CosineKernel, RandomHyperparameterKernel):
    def __init__(self, min_log10_period: float, max_log10_period: float, **kwargs):
        super().__init__(**kwargs)
        self.min_log10_period = min_log10_period
        self.max_log10_period = max_log10_period

    def sample_hyperparameters(self):
        # Sample period.
        log10_period = (
            torch.rand(()) * (self.max_log10_period - self.min_log10_period)
            + self.min_log10_period
        )

        period = 10.0**log10_period
        self.period_length = period
