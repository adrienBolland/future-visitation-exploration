"""
Code inspired from clean rl
https://docs.cleanrl.dev/rl-algorithms/sac/
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
"""
import gymnasium
import torch
import wandb

import numpy as np

from copy import deepcopy
from minigrid.wrappers import ReseedWrapper

from . import utils
from . import samplers
from .buffers import TransitionsReplayBuffer
from .environments import TerminalStateWrapper


def sac_visitation(
        env: gymnasium.Env,
        agent: any,
        intrinsic: any,
        config: dict):
    # initialize the wandb log
    wandb.init(
        project="state-density-rl",
        config=config
    )
    # parameters
    gamma_policy = config.get("gamma_policy")
    gamma_state_dist = config.get("gamma_state_dist")

    # bootstrap horizon
    bootstrap_horizon = config.get("bootstrap_horizon", 1)

    # device
    device = torch.device(config.get("device", "cpu"))

    # scaling
    bool_actor_scale = config.get("bool_actor_scale", False)
    bool_reward_scale = config.get("bool_reward_scale", False)

    # create target agents
    agent_target = deepcopy(agent)

    # initialize the optimizers
    if config.get("nb_offline_visitation_updates"):
        visitation_optimizer = torch.optim.Adam(agent.visitation_model.parameters(),
                                                lr=config.get("lr_state_distribution"))
    actor_optimizer = torch.optim.Adam(agent.actor_model.parameters(), lr=config.get("lr_policy"))
    critic_optimizer = torch.optim.Adam(agent.critic_model.parameters(), lr=config.get("lr_critic"))

    # entropy parameters
    alpha_reward = config.get("alpha_reward", 1.)
    alpha_sac = config.get("alpha_sac", 0.)
    alpha_infty = config.get("alpha_infty", 0.)

    # initialize the replay buffer
    rb = TransitionsReplayBuffer(
        config.get("buffer_max_size"),
        int(np.array(env.observation_space.nvec).sum()),
        int(np.array(env.action_space.n).prod()),
        bootstrap_horizon,
        device,
    )

    # uniform visitation log prob
    uniform_log_prob = np.log(np.array(env.observation_space.nvec)[1:3].prod())
    max_log_prob = np.log(np.array(env.observation_space.nvec)[1:3].prod() * 10**3)

    # initialize the transition sampler
    sampler = samplers.TransitionSamplerIt(env, agent)
    time_step = 0
    for obs, action, reward, next_obs, terminated, truncated, info in sampler.sample(config.get("buffer_max_size")):
        rb.add(obs, next_obs, action, reward, terminated, info, time_step)

        if terminated or truncated:
            time_step = 0
        else:
            time_step += 1

    # initialize control environments and trajectory samplers
    copy_env_1 = deepcopy(env)
    copy_env_1 = ReseedWrapper(copy_env_1)

    copy_env_2 = deepcopy(env)
    copy_env_2 = TerminalStateWrapper(copy_env_2)
    copy_env_2 = gymnasium.wrappers.TimeLimit(copy_env_2, config.get("trajectory_control_len", 200))
    control_sampler = samplers.TrajectorySampler(copy_env_2, agent)

    for it in range(config.get("nb_iteration")):
        # initialize a log dict
        to_log = dict()

        # sample transitions and add to the buffer
        for obs, action, reward, next_obs, terminated, truncated, info in sampler.sample(config.get("nb_transitions_per_step")):
            rb.add(obs, next_obs, action, reward, terminated, info, time_step)

            if terminated or truncated:
                time_step = 0
            else:
                time_step += 1

        for it_off in range(config.get("nb_offline_visitation_updates")):
            # sample a batch of transitions from the buffer
            sa = rb.sample(config.get("batch_size"))
            visitation_loss = intrinsic.visitation_loss(sa)

            # update the visitation model
            visitation_optimizer.zero_grad()
            visitation_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.visitation_model.parameters(),
                                           config.get("visitation_grad_clip", float("inf")))
            visitation_optimizer.step()

            # add to the loss to log
            to_log[f"visitation-loss-{it_off}"] = visitation_loss.item()
            to_log[f"visitation-grad-{it_off}"] = utils.gradient_norm(agent.visitation_model.parameters())

        # update the critic
        for it_off in range(config.get("nb_offline_critic_updates")):
            # sample a batch of transitions from the buffer
            sa = rb.sample(config.get("batch_size"))

            # get the first elements of the transitions
            obs = sa.observations[:, 0, :]
            actions = sa.actions[:, 0, :]
            next_obs = sa.next_observations[:, 0, :]
            rewards = sa.rewards[:, 0, :]
            dones = sa.dones[:, 0, :]

            # compute the target q-value
            with torch.no_grad():
                # sample actions in the next state
                next_actions = agent.actor_model(next_obs).sample()
                next_actions_log_prob = agent.actor_model(next_obs).log_prob(next_actions)

                # bootstrap with the target network
                next_q_value = agent_target.critic_model(torch.cat([next_obs, next_actions], dim=-1))
                next_q_value = (1 - dones.flatten()) * (next_q_value.flatten() - alpha_sac * next_actions_log_prob)

                # compute the reward
                pseudo_rewards = alpha_reward * rewards.flatten()

                # compute the pseudo reward using the visitation model
                if config.get("nb_offline_visitation_updates"):
                    delta_log_prob = intrinsic.intrinsic_reward(sa)
                    pseudo_rewards += alpha_infty * torch.clamp(delta_log_prob - uniform_log_prob, max=max_log_prob)

                    # log
                    to_log[f"rb-intrinsic-rewards-{it_off}"] = -alpha_infty * delta_log_prob.mean().item()

                # log
                to_log[f"rb-rewards-{it_off}"] = sa.rewards.mean().item()
                to_log[f"rb-soft-rewards-{it_off}"] = -next_actions_log_prob.mean().item()

                # compute the soft q-target
                pseudo_rewards = (pseudo_rewards - int(bool_reward_scale) * pseudo_rewards.mean())
                q_target = pseudo_rewards + gamma_policy * next_q_value

            # compute the TD loss of the q model
            q_value = agent.critic_model(torch.cat([obs, actions], dim=-1)).flatten()
            q_loss = torch.nn.functional.mse_loss(q_value, q_target)

            # optimize the critic model
            critic_optimizer.zero_grad()
            q_loss.backward()
            critic_optimizer.step()

            # add to the loss to log
            to_log[f"critic-loss-{it_off}"] = q_loss.item()
            to_log[f"critic-grad-{it_off}"] = utils.gradient_norm(agent.critic_model.parameters())

        # update the policy
        # sample a batch of transitions from the buffer
        sa = rb.sample(config.get("batch_size"))
        obs = sa.observations[:, 0, :]
        sa_weights = sa.time_steps[:, 0]

        with torch.no_grad():
            # sample an action
            actions = agent.actor_model(obs).sample()

            # compute the q-value
            q_value = agent.critic_model(torch.cat([obs, actions], dim=-1)).flatten()

            # compute the weights
            # sa_weight_scale = gamma_policy ** sa_weights.min()
            sa_weights = gamma_policy ** (sa_weights.flatten() - sa_weights.min())

            # sa_weight_scale *= sa_weights.mean()  #  multiply by 'actor_loss' if scaling needed
            sa_weights /= sa_weights.mean()
        
        # apply log trick for the expectation estimation, and let the gradient flow through the entropy
        action_batch_log_prob = agent.actor_model(obs).log_prob(actions)

        soft_q = q_value - alpha_sac * action_batch_log_prob.detach()

        advantage_weight = soft_q - int(bool_actor_scale) * soft_q.mean()
        actor_loss = -(sa_weights * action_batch_log_prob * advantage_weight).mean()

        # optimize the actor model
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # add loss to log
        to_log["soft-q"] = soft_q.mean().item()
        to_log["actor-loss"] = actor_loss.item()
        to_log["actor-grad"] = utils.gradient_norm(agent.actor_model.parameters())

        # update the critic target network
        tau_critic = config.get("target_tau_critic", 1.)
        for param, target_param in zip(agent.critic_model.parameters(), agent_target.critic_model.parameters()):
            target_param.data.copy_(tau_critic * param.data + (1 - tau_critic) * target_param.data)

        # update the visitation target network
        tau_visitation = config.get("target_tau_visitation", 1.)
        for param, target_param in zip(agent.visitation_model.parameters(), agent_target.visitation_model.parameters()):
            target_param.data.copy_(tau_visitation * param.data + (1 - tau_visitation) * target_param.data)

        # control log
        if not it % config.get("control_period", 200):
            wandb_log = dict({"algo-it": it,
                              "critic-mean-loss": np.mean([val for name, val in to_log.items()
                                                           if name.startswith("critic-loss-")]),
                              "critic-mean-grad": np.mean([val for name, val in to_log.items()
                                                           if name.startswith("critic-grad-")]),
                              "soft-q": to_log["soft-q"],
                              "actor-loss": to_log["actor-loss"],
                              "actor-grad": to_log["actor-grad"],
                              "rb-mean-rewards": np.mean([val for name, val in to_log.items()
                                                          if name.startswith("rb-rewards-")]),
                              "rb-mean-soft-rewards": np.mean([val for name, val in to_log.items()
                                                               if name.startswith("rb-soft-rewards-")])})

            if config.get("nb_offline_visitation_updates"):
                wandb_log.update(
                    dict({"visitation-mean-loss": np.mean([val for name, val in to_log.items()
                                                           if name.startswith("visitation-loss-")]),
                          "visitation-mean-grad": np.mean([val for name, val in to_log.items()
                                                           if name.startswith("visitation-grad-")]),
                          "rb-mean-intrinsic-rewards": np.mean([val for name, val in to_log.items()
                                                                if name.startswith("rb-intrinsic-rewards-")])}))

            control_trajectory_len = config.get("trajectory_control_len", 200)
            control_batch_size = config.get("control_batch_size", 64)
            discount_vector = gamma_policy ** torch.arange(control_trajectory_len).view(1, -1)
            state_seq_batch, action_seq_batch, reward_seq_batch, _ = control_sampler.sample(control_batch_size,
                                                                                            control_trajectory_len)

            wandb_log.update(dict({"state std": state_seq_batch.std(dim=1).mean().item(),
                                   "action std": action_seq_batch.std(dim=1).mean().item(),
                                   "expected return": torch.mean(torch.sum(discount_vector * reward_seq_batch, dim=1)).item()}))
            wandb_log.update(utils.state_joint_entropy_mc(state_seq_batch, gamma_state_dist))
            wandb_log.update(utils.state_joint_entropy_mc(agent.feature_map(state_seq_batch), gamma_state_dist, "-feature"))

            if config.get("verbose", False):
                print(it)
                for n, v in wandb_log.items():
                    print(n, " : ", v)

            density_plot_period = config.get("density_plot_period", -1)
            if density_plot_period > 0 and not it % density_plot_period:
                expected_return = utils.render_occupancy(copy_env_1, agent, config, config.get("save_path") + f"-{it}-")
                wandb_log.update(dict({"expected return variable": expected_return}))

            wandb.log(wandb_log)
