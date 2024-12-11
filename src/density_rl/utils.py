import torch
import numpy as np

from minigrid.wrappers import ReseedWrapper

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from . import samplers


def gradient_norm(parameters):
    total_norm = 0.
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm


def state_entropy_mc(env, agent, gamma):
    nb_mc_trajectories = 150
    trajectory_len = 200

    # sample trajectories
    env = ReseedWrapper(env)
    state_seq_batch, _, _, _ = samplers.TrajectorySampler(env, agent).sample(nb_mc_trajectories, trajectory_len)

    # get the state shape
    state_dim = env.observation_space.nvec.tolist()
    # state_dim = [env.observation_space.n.tolist()]

    dict_marginal_entropy = state_marginal_entropy_mc(state_seq_batch, state_dim, gamma)
    dict_joint_entropy = state_joint_entropy_mc(state_seq_batch, gamma)

    return dict_marginal_entropy | dict_joint_entropy


def state_marginal_entropy_mc(state_seq_batch, state_dim, gamma):
    # compute the discount
    trajectory_len = state_seq_batch.shape[1]
    discount_vector = torch.tensor(gamma) ** torch.linspace(0, trajectory_len - 1, trajectory_len).view(1, -1, 1)
    scale = torch.sum(discount_vector)

    # count the state realization
    state_weighted_count = torch.sum(torch.flatten(state_seq_batch * discount_vector / scale, 0, 1), dim=0)
    state_count = torch.sum(torch.flatten(state_seq_batch, 0, 1), dim=0)

    # compute the mean marginal entropy
    state_weighted_entropy = torch.empty(len(state_dim))
    state_entropy = torch.empty(len(state_dim))

    for i, (s_w_count, s_count) in enumerate(zip(torch.split(state_weighted_count, state_dim),
                                                 torch.split(state_count, state_dim))):
        s_w_proba = s_w_count / s_w_count.sum()
        s_proba = s_count / s_count.sum()
        state_weighted_entropy[i] = -torch.nan_to_num(s_w_proba * torch.log(s_w_proba)).sum()
        state_entropy[i] = -torch.nan_to_num(s_proba * torch.log(s_proba)).sum()

    # compute the mean weighted and scaled entropy
    mean_weighted_entropy = state_weighted_entropy.mean()
    mean_entropy = state_entropy.mean()

    state_dim = torch.tensor(state_dim)
    mean_weighted_normalized_entropy = (state_weighted_entropy / state_dim).mean()
    mean_normalized_entropy = (state_entropy / state_dim).mean()

    return dict({"mean-weighted-entropy": mean_weighted_entropy.item(),
                 "mean-entropy": mean_entropy.item(),
                 "mean-weighted-normalized-entropy": mean_weighted_normalized_entropy.item(),
                 "mean-normalized-entropy": mean_normalized_entropy.item()})


def state_joint_entropy_mc(state_seq_batch, gamma, suffix=""):
    # encode the states with a unique numeral value
    code = 2 ** torch.linspace(0, state_seq_batch.shape[2] - 1, state_seq_batch.shape[2], dtype=torch.long)

    dims_code = list(range(-len(code.shape), 0))
    int_state_seq_batch = torch.tensordot(state_seq_batch.long(), code, dims=(dims_code, dims_code))

    # count the state occurrence
    state_weighted_count = dict()
    state_count = dict()

    state_weighted_sum = 0.
    state_sum = 0.

    for b in range(state_seq_batch.shape[0]):
        for t in range(state_seq_batch.shape[1]):
            int_state = int_state_seq_batch[b, t].item()

            if int_state not in state_weighted_count:
                state_weighted_count[int_state] = 0.
                state_count[int_state] = 0.

            state_weighted_count[int_state] += gamma ** t
            state_count[int_state] += 1.

            state_weighted_sum += gamma ** t
            state_sum += 1.

    # compute the entropy based on the counts
    joint_state_weighted_entropy = 0.
    joint_state_entropy = 0.

    for int_state in state_weighted_count:
        weighted_p = state_weighted_count[int_state] / state_weighted_sum
        p = state_count[int_state] / state_sum

        joint_state_weighted_entropy -= weighted_p * np.log(weighted_p)
        joint_state_entropy -= p * np.log(p)

    return dict({"joint-weighted-entropy" + suffix: joint_state_weighted_entropy,
                 "joint-entropy" + suffix: joint_state_entropy})


def render_occupancy(env, agent, config, name=None):
    gamma = config.get("gamma_state_dist")

    # simulation parameters
    nb_mc_trajectories = 500
    trajectory_len = config.get("time_limit", 200)

    # initialize the transition sampler
    sampler = samplers.TransitionSamplerIt(env, agent)

    # sample trajectories  trajectories
    states = []
    weight = []

    t = 0
    trajectory_id = 0
    cum_sum = 0.
    cum_reward = 0.
    for obs, _, reward, next_obs, terminated, truncated, _ in sampler.sample(nb_mc_trajectories * trajectory_len):

        if trajectory_id >= nb_mc_trajectories:
            break

        proba = (1 - gamma) * gamma**t
        cum_sum += proba

        cum_reward += gamma**t * reward

        states.append(obs)
        weight.append(proba)

        if terminated or truncated:
            states.append(next_obs)
            weight.append(1 - cum_sum)

            t = 0
            trajectory_id += 1
            cum_sum = 0.
        else:
            t += 1

    # scale cum reward
    cum_reward /= nb_mc_trajectories

    # change list to arrays
    states = np.stack(states)
    weight = np.stack(weight)

    # get the position in grid format
    if config.get("xy_model", False):
        nb_xy = env.observation_space.nvec.tolist()[1]
        nb_x_y = int(np.sqrt(nb_xy))
        xy_code =  np.split(states, np.cumsum(env.observation_space.nvec.tolist()), -1)[1]
        xy_number = np.sum(xy_code * np.arange(nb_xy).reshape(1, -1), axis=-1).astype(int)
        xy = np.zeros((states.shape[0], nb_x_y, nb_x_y))
        xy[np.arange(states.shape[0]), xy_number % nb_x_y, xy_number // nb_x_y] = 1
    else:
        x, y = np.split(states, np.cumsum(env.observation_space.nvec.tolist()), -1)[1:3]
        xy = np.expand_dims(x, -1) * np.expand_dims(y, -2)

    # compute the discounted state proba
    xy_weighted_count = np.sum(xy * weight.reshape((-1, 1, 1)), axis=0) + 1e-11
    xy_weighted_prob = xy_weighted_count / xy_weighted_count.sum()

    plt.figure()
    plt.imshow(xy_weighted_prob, cmap='hot', interpolation='nearest', norm=LogNorm())
    plt.colorbar()
    plt.clim(1e-1, 1e-10)

    if name is None:
        plt.show()
    else:
        plt.savefig(name + f"disc-occupancy.pdf")
        plt.close()

    # compute the stationary state proba
    xy_count = np.sum(xy, axis=0) + 1e-11
    xy_prob = xy_count / xy_count.sum()

    plt.figure()
    plt.imshow(xy_prob, cmap='hot', interpolation='nearest', norm=LogNorm())
    plt.colorbar()
    plt.clim(1e-1, 1e-10)

    if name is None:
        plt.show()
    else:
        plt.savefig(name + f"marginal-occupancy.pdf")
        plt.close()

    return cum_reward