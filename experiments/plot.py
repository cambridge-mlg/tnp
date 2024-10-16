import copy
import os
from typing import Callable, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from tnp.data.base import Batch
from tnp.data.synthetic import SyntheticBatch
from tnp.models.base import LatentNeuralProcess
from tnp.utils.np_functions import np_pred_fn

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot(
    model: Union[
        nn.Module,
        Callable[..., torch.distributions.Distribution],
    ],
    batches: List[Batch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    x_range: Tuple[float, float] = (-4.0, 4.0),
    y_lim: Tuple[float, float] = (-3.0, 3.0),
    points_per_dim: int = 64,
    name: str = "plot",
    savefig: bool = False,
    logging: bool = True,
    pred_fn: Callable = np_pred_fn,
):
    steps = int(points_per_dim * (x_range[1] - x_range[0]))

    x_plot = torch.linspace(x_range[0], x_range[1], steps).to(batches[0].xc)[
        None, :, None
    ]
    for i in range(num_fig):
        batch = batches[i]
        for key, value in vars(batch).items():
            if isinstance(value, torch.Tensor):
                setattr(batch, key, value[:1])

        plot_batch = copy.deepcopy(batch)
        plot_batch.xt = x_plot

        with torch.no_grad():
            y_plot_pred_dist = pred_fn(model, plot_batch, num_samples=20)
            yt_pred_dist = pred_fn(model, batch, num_samples=20)

        model_nll = -yt_pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        mean, std = y_plot_pred_dist.mean, y_plot_pred_dist.stddev

        # Make figure for plotting
        fig = plt.figure(figsize=figsize)

        # Plot context and target points
        plt.scatter(
            batch.xc[0, :, 0].cpu(),
            batch.yc[0, :, 0].cpu(),
            c="k",
            label="Context",
            s=30,
        )

        plt.scatter(
            batch.xt[0, :, 0].cpu(),
            batch.yt[0, :, 0].cpu(),
            c="r",
            label="Target",
            s=30,
        )

        # Plot model predictions
        plt.plot(
            x_plot[0, :, 0].cpu(),
            mean[0, :, 0].cpu(),
            c="tab:blue",
            lw=3,
        )

        plt.fill_between(
            x_plot[0, :, 0].cpu(),
            mean[0, :, 0].cpu() - 2.0 * std[0, :, 0].cpu(),
            mean[0, :, 0].cpu() + 2.0 * std[0, :, 0].cpu(),
            color="tab:blue",
            alpha=0.2,
            label="Model",
        )

        # Plot samples if neural process.
        if isinstance(model, LatentNeuralProcess):
            samples = y_plot_pred_dist.component_distribution.base_dist.loc
            for sample_idx in range(samples.shape[1]):
                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    samples[0, sample_idx, :, 0].cpu(),
                    c="tab:blue",
                    lw=1,
                    alpha=0.5,
                    label="Samples" if sample_idx == 0 else None,
                )

        title_str = f"$N = {batch.xc.shape[1]}$ NLL = {model_nll:.3f}"

        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            with torch.no_grad():
                gt_mean, gt_std, _ = batch.gt_pred(
                    xc=batch.xc,
                    yc=batch.yc,
                    xt=x_plot,
                )
                _, _, gt_loglik = batch.gt_pred(
                    xc=batch.xc,
                    yc=batch.yc,
                    xt=batch.xt,
                    yt=batch.yt,
                )
                gt_nll = -gt_loglik.sum() / batch.yt[..., 0].numel()

            # Plot ground truth
            plt.plot(
                x_plot[0, :, 0].cpu(),
                gt_mean[0, :].cpu(),
                "--",
                color="tab:purple",
                lw=3,
            )

            plt.plot(
                x_plot[0, :, 0].cpu(),
                gt_mean[0, :].cpu() + 2 * gt_std[0, :].cpu(),
                "--",
                color="tab:purple",
                lw=3,
            )

            plt.plot(
                x_plot[0, :, 0].cpu(),
                gt_mean[0, :].cpu() - 2 * gt_std[0, :].cpu(),
                "--",
                color="tab:purple",
                label="Ground truth",
                lw=3,
            )

            title_str += f" GT NLL = {gt_nll:.3f}"

        plt.title(title_str, fontsize=24)
        plt.grid()

        # Set axis limits
        plt.xlim(x_range)
        plt.ylim(y_lim)

        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        plt.legend(loc="upper right", fontsize=20)
        plt.tight_layout()

        fname = f"fig/{name}/{i:03d}"
        if wandb.run is not None and logging:
            wandb.log({fname: wandb.Image(fig)})
        elif savefig:
            if not os.path.isdir(f"fig/{name}"):
                os.makedirs(f"fig/{name}")
            plt.savefig(fname, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
