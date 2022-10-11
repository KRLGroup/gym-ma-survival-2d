from typing import Optional, List
import time
import os

import pygame

import numpy as np
import gym
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

from masurvival.envs.masurvival_env import MaSurvival
from sb3_experiments.vec_env_costume import VecEnvCostume
from sb3_experiments.sb3_models import MhSaExtractor
from sb3_experiments.sb3_log_stats import LogStats

import experiments


def main(exp_dirpath):
    print(f'Loading experiment at {exp_dirpath}.')
    exp = experiments.get_experiment(exp_dirpath)
    # Make sure all experiment directories exist etc.
    experiments.init_experiment(exp)
    env = VecEnvCostume(MaSurvival(config=exp['env_config']))
    prev_run_ids = experiments.get_run_ids(exp)
    run = None # assigned in the next if
    if len(prev_run_ids) == 0:
        print('No previous runs found.')
        run = experiments.get_run(exp, 1)
        policy_kwargs = dict(
            features_extractor_class=MhSaExtractor,
            features_extractor_kwargs=dict(
                entity_keys=env.env.entity_keys(),
            )
        )
        model = PPO(
            MultiInputActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=exp['tb_log_dir'],
            n_steps=2048*8,
            batch_size=64*8,
            n_epochs=50*8,
            target_kl=0.01,
        )
    else:
        next_run_id = max(prev_run_ids) + 1
        run = experiments.get_run(exp, next_run_id)
        last_run = experiments.get_run(exp, next_run_id-1)
        print(f'Loading state from the last run: {last_run["name"]}.')
        model = PPO.load(last_run['model'], env=env)

    experiments.init_run(run)
    print(f'This is run {run["name"]}.')

    log_stats = LogStats()
    checkpoint_callback = CheckpointCallback(
        # save freq is doubled for some sb3 reason
        save_freq=500_000, save_path=run['checkpoints_dir'],
        name_prefix=exp['checkpoint_prefix'],
    )
    
    try:
        model.learn(
            total_timesteps=10_000_000,
            reset_num_timesteps=True,
            callback=[log_stats, checkpoint_callback],
        )
    except KeyboardInterrupt:
        pass
    
    print(f'Saving model file.')
    model.save(run['model'])



# Script stuff

argparse_desc = \
"""Train the model for the given experiment.
"""

argparse_args = [
    (['exp_dirpath'], {
        'metavar': 'PATH',
        'type': str,
        'help': 'The path to the experiment directory.',
    }),
#     (['-t', '--tags'], {
#         'required': True,
#         'metavar': 'T',
#         'type': str,
#         'nargs': '+',
#         'help': 'Tags to extract from the tensorboard runs.',
#     }),
#     (['-o', '--output'], {
#         'dest': 'out_fpath',
#         'metavar': 'OUTPATH',
#         'default': None,
#         'help': 'The path to save the concatenated dataframe to.',
#     }),
]

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=argparse_desc)
    for args, kwargs in argparse_args:
        argparser.add_argument(*args, **kwargs)
    cli_args = argparser.parse_args()
    #print(vars(cli_args))
    main(**vars(cli_args))


