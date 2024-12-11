import gymnasium

import numpy as np

from minigrid.core.world_object import Wall
from gymnasium.envs.registration import register

from density_rl.environments import ActionOHEWrapper, StateMultiOHEWrapper
from density_rl.intrinsic_future_visitation import FutureVisitationReward
from density_rl.intrinsic_marginal_visitation import MarginalVisitationReward
from density_rl.minigrid_agents import MinigridPositionActorCriticVisitationAgent
from density_rl.minigrid_wrappers import MinigridXYModelWrapper
from density_rl.models import (CategoricalModel, MultiCategoricalModel, ForwardModel, MultiCategoricalMarginalModel,
                               ActorCriticVisitationAgent)
from density_rl import minigrid_wrappers

def _register_custom_minigrid():
    register(
        id="MiniGrid-SimpleCrossingS11N1-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 11, "num_crossings": 1, "obstacle_type": Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS13N1-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 13, "num_crossings": 1, "obstacle_type": Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS15N1-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 15, "num_crossings": 1, "obstacle_type": Wall},
    )


def get_minigrid_terminal(config, render_mode=None):
    _register_custom_minigrid()
    env_name = config.get("env_name")

    env_dict = dict({"MiniGrid-Empty-": minigrid_wrappers.EmptyMinigridMultiDiscreteWrapper,
                     "MiniGrid-SimpleCrossing": minigrid_wrappers.CrossingS9N1MinigridMultiDiscreteWrapper,
                     "MiniGrid-DoorKey-": minigrid_wrappers.DoorkeyMinigridMultiDiscreteWrapper,
                     "MiniGrid-FourRooms": minigrid_wrappers.FourRoomsMultiDiscreteWrapper})

    for name_reference, MinigridWrapper in env_dict.items():
        if name_reference in env_name:
            env = gymnasium.make(env_name, render_mode=render_mode)
            env = gymnasium.wrappers.TransformReward(env, lambda r: float(r > 0.))
            env = MinigridWrapper(env)
            if config.get("xy_model", False):
                env = MinigridXYModelWrapper(env)
            env = ActionOHEWrapper(env)
            env = StateMultiOHEWrapper(env)
            env = gymnasium.wrappers.TimeLimit(env, config.get("time_limit", 200))
            env.reset()

            return env

    raise NotImplementedError()


def get_sac_future_visitation_agent(env, config):
    nb_flat_states = int(np.array(env.observation_space.nvec).sum())
    nb_discrete_states = env.observation_space.nvec
    nb_flat_actions = int(np.array(env.action_space.n).prod())

    # initialize the policy, critic and visitation models
    policy_model = CategoricalModel(condition_size=nb_flat_states,
                                    nb_classes=nb_flat_actions,
                                    layers=config.get("policy_layers"))

    condition_size = nb_flat_states + nb_flat_actions
    critic_model = ForwardModel(condition_size=condition_size,
                                nb_outputs=1,
                                layers=config.get("critic_layers"))

    visitation_model = MultiCategoricalModel(condition_size=condition_size,
                                             vect_classes=nb_discrete_states,
                                             layers=config.get("visitation_layers"))

    # initialize the agent
    # agent = ActorCriticVisitationAgent(policy_model, critic_model, visitation_model)
    agent = MinigridPositionActorCriticVisitationAgent(policy_model, critic_model, visitation_model, env,
                                                       xy_model=config.get("xy_model", False))

    # initialize the intrinsic return
    intrinsic = FutureVisitationReward(agent, config.get("gamma_state_dist"))

    return agent, intrinsic


def get_sac_marginal_visitation_agent(env, config):
    nb_flat_states = int(np.array(env.observation_space.nvec).sum())
    nb_discrete_states = env.observation_space.nvec
    nb_flat_actions = int(np.array(env.action_space.n).prod())

    # initialize the policy, critic and visitation models
    policy_model = CategoricalModel(condition_size=nb_flat_states,
                                    nb_classes=nb_flat_actions,
                                    layers=config.get("policy_layers"))

    condition_size = nb_flat_states + nb_flat_actions
    critic_model = ForwardModel(condition_size=condition_size,
                                nb_outputs=1,
                                layers=config.get("critic_layers"))

    visitation_model = MultiCategoricalMarginalModel(vect_classes=nb_discrete_states)

    # initialize the agent
    # agent = ActorCriticVisitationAgent(policy_model, critic_model, visitation_model)
    agent = MinigridPositionActorCriticVisitationAgent(policy_model, critic_model, visitation_model, env,
                                                       xy_model=config.get("xy_model", False))

    # initialize the intrinsic return
    intrinsic = MarginalVisitationReward(agent, config.get("gamma_state_dist"))

    return agent, intrinsic