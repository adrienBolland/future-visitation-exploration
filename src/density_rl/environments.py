import numpy as np

from copy import deepcopy
from typing import Any, SupportsFloat

from gymnasium import Env
from gymnasium.core import ActType, Wrapper, ObservationWrapper, ActionWrapper, ObsType, WrapperObsType, WrapperActType
from gymnasium.spaces import Discrete


class TerminalStateWrapper(Wrapper):
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        self._terminal_state = None
        self._nb_step_calls = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        # reset the variables
        self._nb_step_calls = 0
        self._terminal_state = None

        # call the environment reset method
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # increase call counter
        self._nb_step_calls += 1

        if self._terminal_state is None:
            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated:
                self._terminal_state = deepcopy(observation)
        else:
            observation = deepcopy(self._terminal_state)
            reward = 0.
            max_steps = self.env.spec.max_episode_steps
            truncated = False if max_steps is None else self._nb_step_calls >= max_steps
            info = dict()

        return observation, reward, False, truncated, info


class StateOHEWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation: ObsType):
        ohe_observation = np.zeros((self.observation_space.n,))
        ohe_observation[observation] = 1.
        return ohe_observation


class StateMultiOHEWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation: ObsType):
        ohe_falt_observations = []
        for i, n in enumerate(self.observation_space.nvec):
            ohe_observation = np.zeros((n,))
            ohe_observation[observation[..., i]] = 1.
            ohe_falt_observations.append(ohe_observation)
        return np.concatenate(ohe_falt_observations, axis=-1)


class ActionOHEWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.ohe_code = np.linspace(0, self.action_space.n - 1, self.action_space.n)

    def action(self, action: ObsType):
        return np.dot(action, self.ohe_code)


class DiscreteGridEnv(Env):
    def __init__(self):
        super().__init__()

        # set the size of the grid
        self._grid_size = 7
        self._max_horizon = 1000

        # initialize the state and action space
        self.action_space = Discrete(5)
        self.observation_space = Discrete(self._grid_size * self._grid_size)

        # internal state
        self._state = None
        self._step = None
        self._wall = True

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self._state = np.array([0, (self._grid_size - 1) / 2])
        self._step = 0

        return self._int_state, dict()

    def step(self, action: ActType):

        x, y = self._state
        if (action == 0
                or (self._wall and
                    ((action == 1 and x + 1 == self._grid_size // 2 and y != self._grid_size - 1)
                     or (action == 3 and x - 1 == self._grid_size // 2 and y != self._grid_size - 1)
                     or (action == 4 and x == self._grid_size // 2 and y == self._grid_size - 1)))):
            ds = np.array([0., 0.])
        elif action == 1:
            ds = np.array([1., 0.])
        elif action == 2:
            ds = np.array([0., 1.])
        elif action == 3:
            ds = np.array([-1., 0.])
        elif action == 4:
            ds = np.array([0., -1.])
        else:
            raise Exception("action error")

        self._state = np.clip(self._state + ds, 0., self._grid_size - 1)
        self._step += 1

        return self._int_state, int(self._int_state == self._int_target), False, self._step >= self._max_horizon, dict()

    @property
    def _int_state(self):
        return int(self._grid_size * self._state[0] + self._state[1])

    @property
    def _int_target(self):
        return self._grid_size * (self._grid_size - 1)


class ContinuousGridEnv(Env):
    def __init__(self):
        pass

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        pass

    def step(self, action: ActType):
        pass
