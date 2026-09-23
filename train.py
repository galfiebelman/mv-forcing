# Derived from Self-Forcing (https://github.com/guandeh17/Self-Forcing, commit 33593df), licensed under the
# Apache License 2.0. Modified for MV-Forcing.
# See NOTICE.md and licenses/LICENSE.Apache-2.0.

import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import ODETrainer, ScoreDistillationTrainerMV, ScoreDistillationTrainerMVRecam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=False)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/base.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    # config.logdir = args.logdir
    # config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    if config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation_mv":
        trainer = ScoreDistillationTrainerMV(config)
    elif config.trainer == "score_distillation_mv_recam":
        trainer = ScoreDistillationTrainerMVRecam(config)
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
