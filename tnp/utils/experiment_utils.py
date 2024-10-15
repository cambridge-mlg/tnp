import argparse
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import hiyapyco
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb

from .lightning_utils import LitWrapper


def create_default_config() -> DictConfig:
    default_config = {
        "misc": {
            "resume_from_checkpoint": None,
            "logging": True,
            "seed": 0,
            "plot_interval": 1,
            "lightning_eval": True,
            "num_plots": 5,
            "gradient_clip_val": 0.5,
            "only_plots": False,
            "savefig": False,
            "subplots": True,
            "loss_fn": {
                "_target_": "tnp.utils.np_functions.np_loss_fn",
                "_partial_": True,
            },
            "pred_fn": {
                "_target_": "tnp.utils.np_functions.np_pred_fn",
                "_partial_": True,
            },
            "num_workers": 1,
            "num_val_workers": 1,
            "log_interval": 10,
            "checkpoint_interval": 1,
            "check_val_every_n_epoch": 1,
        }
    }
    return OmegaConf.create(default_config)


def extract_config(
    raw_config: Union[str, Dict],
    config_changes: Optional[List[str]] = None,
    combine_default: bool = True,
) -> Tuple[DictConfig, Dict]:
    """Extract the config from the config file and the config changes.

    Arguments:
        config_file: path to the config file.
        config_changes: list of config changes.

    Returns:
        config: config object.
        config_dict: config dictionary.
    """
    # Register eval.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    if isinstance(raw_config, str):
        config = OmegaConf.load(raw_config)
    else:
        config = OmegaConf.create(raw_config)

    if combine_default:
        default_config = create_default_config()
        config = OmegaConf.merge(default_config, config)

    config_changes = OmegaConf.from_cli(config_changes)
    config = OmegaConf.merge(config, config_changes)
    config_dict = OmegaConf.to_container(config, resolve=True)

    return config, config_dict


def deep_convert_dict(layer: Any):
    to_ret = layer
    if isinstance(layer, OrderedDict):
        to_ret = dict(layer)

    try:
        for key, value in to_ret.items():
            to_ret[key] = deep_convert_dict(value)
    except AttributeError:
        pass

    return to_ret


def initialize_experiment() -> DictConfig:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="+")
    args, config_changes = parser.parse_known_args()

    raw_config = deep_convert_dict(
        hiyapyco.load(
            args.config,
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )

    # Initialise experiment, make path.
    config, _ = extract_config(raw_config, config_changes)

    # Instantiate experiment and load checkpoint.
    pl.seed_everything(config.misc.seed)
    experiment = instantiate(config)
    experiment.config = config
    pl.seed_everything(experiment.misc.seed)

    return experiment


def initialize_evaluation() -> DictConfig:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, help="e.g. user/project/run")
    parser.add_argument("--config", type=str, nargs="+", help="e.g. config.yaml")
    parser.add_argument("--checkpoint", type=str, help="e.g. users/project/artifact")
    args, config_changes = parser.parse_known_args()

    raw_config = deep_convert_dict(
        hiyapyco.load(
            args.config,
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )

    # Initialise evaluation, make path.
    config, _ = extract_config(raw_config, config_changes)

    # Initialise wandb.
    api = wandb.Api()
    run = api.run(args.run_path)
    run = wandb.init(
        resume="allow",
        project=run.project,
        name=run.name,
        id=run.id,
    )

    # Instantiate.
    pl.seed_everything(config.misc.seed)
    experiment = instantiate(config)
    pl.seed_everything(config.misc.seed)

    # Load checkpoint from run artifact.
    artifact = run.use_artifact(args.checkpoint)
    artifact_dir = artifact.download()
    ckpt_file = os.path.join(artifact_dir, "model.ckpt")

    ckpt = torch.load(ckpt_file, map_location="cpu")
    print(f"Checkpoint epochs: {ckpt['epoch']}")

    # Load in the checkpoint.
    experiment.lit_model = (
        LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
            checkpoint_path=ckpt_file,
            map_location="cpu",
            strict=True,
        )
    )

    return experiment
