import os
import warnings
import argparse
import yaml
import torch

import numpy as np

from itertools import batched

from density_rl.maxent_sac_visitation import sac_visitation
from utils_minigrid import get_minigrid_terminal, get_sac_future_visitation_agent, get_sac_marginal_visitation_agent


def sac_exp_minigrid(config):
    # activate wandb
    if not config.get("wandb", True):
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_SILENT"] = "true"

    # initialize the grid environment, agent and intrinsic reward
    env = get_minigrid_terminal(config)

    intrinsic_reward = config.get("intrinsic_reward", "future-visitation")
    if intrinsic_reward == "future-visitation":
        agent, intrinsic = get_sac_future_visitation_agent(env, config)
    elif intrinsic_reward == "marginal-visitation":
        agent, intrinsic = get_sac_marginal_visitation_agent(env, config)
    else:
        raise NotImplementedError

    # add info to config
    config["environment"] = type(env.unwrapped).__name__
    config["actor_model"] = type(agent.actor_model).__name__
    config["visitation_model"] = type(agent.visitation_model).__name__

    # Apply SAC algorithm
    sac_visitation(env=env, agent=agent, intrinsic=intrinsic, config=config)

    # save agent
    torch.save(agent.state_dict(), config.get("save_path"))


if __name__ == "__main__":
    # parse the input
    parser = argparse.ArgumentParser(description="script launching experiments.")
    parser.add_argument("-config", "--config_file", type=str, required=True)
    parser.add_argument("-exp", "--experiment_name", type=str, default="sac_minigrid")
    parser.add_argument("-args", "--arguments", nargs='*', default=[])
    parser.add_argument("-s", "--seed", type=int, default=None)

    parsed_input = parser.parse_args()

    # get config
    config_file = yaml.load(open(parsed_input.config_file), Loader=yaml.FullLoader)

    # add to config files
    for name, value in batched(parsed_input.arguments, 2):
        config_file[name] = value

    # set number of threads
    torch.set_num_threads(config_file.get("nb_threads", 1))

    # set seeds
    seed = parsed_input.seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    with warnings.catch_warnings(action="ignore"):
        # experiments
        if parsed_input.experiment_name == "sac_minigrid":
            sac_exp_minigrid(config=config_file)
        else:
            raise NotImplementedError()
