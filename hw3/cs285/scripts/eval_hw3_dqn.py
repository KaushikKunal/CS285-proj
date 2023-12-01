import time
import argparse

from cs285.agents.dqn_agent import DQNAgent
import cs285.env_configs

import os
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer

from cs285.scripts.scripting_utils import make_logger, make_config, make_fake_logger

MAX_NVIDEO = 2


def run_eval_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )
    agent.load_state_dict(torch.load("./models/dqn_LunarLander-v2_s64_l2_d0.99_doubleqagent.pt"))
    agent.prune(args.prune_amount)
    agent.lra(args.derank_amount)

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 4

    ep_len = env.spec.max_episode_steps

    observation = None

    def reset_env_training():
        nonlocal observation

        observation = env.reset()

        assert not isinstance(
            observation, tuple
        ), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
        observation = np.asarray(observation)

    reset_env_training()

    step = 0
    # Evaluate
    trajectories = utils.sample_n_trajectories(
        eval_env,
        agent,
        args.num_eval_trajectories,
        ep_len,
    )
    returns = [t["episode_statistics"]["r"] for t in trajectories]
    ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

    logger.log_scalar(np.mean(returns), "eval_return", step)
    logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

    if len(returns) > 1:
        logger.log_scalar(np.std(returns), "eval/return_std", step)
        logger.log_scalar(np.max(returns), "eval/return_max", step)
        logger.log_scalar(np.min(returns), "eval/return_min", step)
        logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
        logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
        logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

    if args.num_render_trajectories > 0:
        video_trajectories = utils.sample_n_trajectories(
            render_env,
            agent,
            args.num_render_trajectories,
            ep_len,
            render=True,
        )

        logger.log_paths_as_videos(
            video_trajectories,
            step,
            fps=fps,
            max_videos_to_save=args.num_render_trajectories,
            video_title="eval_rollouts",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--prune_amount", "-pa", type=float, default=0)
    parser.add_argument("--derank_amount", "-lra", type=float, default=0)
    parser.add_argument("--no_log", "-nlog", action="store_true")

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = f"eval_dqn_prune{args.prune_amount}_rank{args.derank_amount}_"

    config = make_config(args.config_file)
    if not args.no_log:
        logger = make_logger(logdir_prefix, config)
    else:
        logger = make_fake_logger()

    run_eval_loop(config, logger, args)


if __name__ == "__main__":
    main()
