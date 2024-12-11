import numpy as np

from gymnasium import ObservationWrapper
from gymnasium.core import ObsType, Env
from gymnasium.spaces import MultiDiscrete
from minigrid.core.constants import OBJECT_TO_IDX

from minigrid.envs import EmptyEnv, CrossingEnv, DoorKeyEnv
from scipy.signal import convolve2d


def _get_position(grid, object_name):
    position = []

    for p, obj in enumerate(grid.grid):
        if obj is not None and obj.type is object_name:
            # transform the position encoding
            x, y = p % grid.height, p // grid.width

            if x == 0 or y == 0 or x == grid.width-1 or y == grid.height-1:
                continue

            position.append(x - 1)
            position.append(y - 1)

    return position

def _get_one_way(grid):
    walls_map = (grid.encode()[:, :, 0] == OBJECT_TO_IDX["wall"]).astype(float)

    v_filter = np.array([[1, 1., 0., 1., 1]])
    h_filter = np.array([[1, 1., 0., 1., 1]]).T

    v_walls = (convolve2d(walls_map, v_filter, mode='same', fillvalue=1) * (1 - walls_map) == 4)[1:-1, 1:-1].astype(float)
    v_walls_x, v_walls_y = np.nonzero(v_walls)
    v_walls_bool = np.zeros_like(v_walls_x)
    v_walls_flat = np.concatenate((v_walls_x, v_walls_y, v_walls_bool), axis=0).flatten()

    h_walls = (convolve2d(walls_map, h_filter, mode='same', fillvalue=1) * (1 - walls_map) == 4)[1:-1, 1:-1].astype(float)
    h_walls_x, h_walls_y = np.nonzero(h_walls)
    h_walls_bool = np.ones_like(h_walls_x)
    h_walls_flat = np.concatenate((h_walls_x, h_walls_y, h_walls_bool), axis=0).flatten()

    way = np.concatenate((v_walls_flat, h_walls_flat)).tolist()

    if len(way) != 3:
        raise NotImplementedError

    return way


class EmptyMinigridMultiDiscreteWrapper(ObservationWrapper):
    def __init__(self, env: EmptyEnv):
        super().__init__(env)

        grid_len = env.grid.height
        self.observation_space = MultiDiscrete((4,) + int(2) * (grid_len - 2,))

    def observation(self, observation: ObsType):
        x, y = self.agent_pos[0] - 1, self.agent_pos[1] - 1
        direction = self.agent_dir
        return np.array([direction, x, y])


class CrossingS9N1MinigridMultiDiscreteWrapper(ObservationWrapper):
    def __init__(self, env: CrossingEnv):
        super().__init__(env)

        grid_len = env.grid.height
        # self.observation_space = MultiDiscrete((4,) + int(2 + 2*(grid_len - 3)) * (grid_len - 2,))
        self.observation_space = MultiDiscrete((4,) + (2 + 2) * (grid_len - 2,) + (2,))

    def observation(self, observation: ObsType):
        x, y = self.agent_pos[0] - 1, self.agent_pos[1] - 1
        direction = self.agent_dir

        # walls = _get_position(self.grid, "wall")
        # return np.array([direction, x, y] + walls)

        ways = _get_one_way(self.grid)

        return np.array([direction, x, y] + ways)


class DoorkeyMinigridMultiDiscreteWrapper(ObservationWrapper):
    def __init__(self, env: DoorKeyEnv):
        super().__init__(env)

        grid_len = env.grid.height
        # self.observation_space = MultiDiscrete((4,) + int(2 + 2*(grid_len - 3)) * (grid_len - 2,) + 2 * (grid_len - 1,))
        self.observation_space = MultiDiscrete((4,) + (2 + 2) * (grid_len - 2,) + (2,) + 2 * (grid_len - 1,))

    def observation(self, observation: ObsType):
        x, y = self.agent_pos[0] - 1, self.agent_pos[1] - 1
        direction = self.agent_dir
        key = [p+1 for p in _get_position(self.grid, "key")]  # reserve zero value when no key
        ways = _get_one_way(self.grid)

        if len(key) == 0:
            key = [0, 0]

        # walls = _get_position(self.grid, "wall")
        # return np.array([direction, x, y] + walls + key)

        return np.array([direction, x, y] + ways + key)


class FourRoomsMultiDiscreteWrapper(ObservationWrapper):
    def __init__(self, env: DoorKeyEnv):
        super().__init__(env)

        grid_len = env.grid.height
        self.observation_space = MultiDiscrete((4,) + int(2 + 2 + 4) * (grid_len - 2,))

        self._target_positions = np.linspace(0, grid_len - 3, grid_len - 2, dtype=np.int32)

    def observation(self, observation: ObsType):
        x, y = self.agent_pos[0] - 1, self.agent_pos[1] - 1
        direction = self.agent_dir
        walls = _get_position(self.grid, "wall")

        walls_positions = np.array(walls).reshape((-1, 2))
        ways_x = np.setdiff1d(self._target_positions, walls_positions[:, 0]).tolist()
        ways_y = np.setdiff1d(self._target_positions, walls_positions[:, 1]).tolist()

        goal = _get_position(self.grid, "goal")

        return np.array([direction, x, y] + goal + ways_x + ways_y)


class MinigridXYModelWrapper(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        self.grid_len = np.mean(env.observation_space.nvec[1:2]).astype(int)
        obs_space_x_y = tuple(env.observation_space.nvec)
        self.observation_space = MultiDiscrete(obs_space_x_y[0:1] + (self.grid_len**2,)  + obs_space_x_y[3:])

    def observation(self, observation: ObsType):
        xy = observation[..., (1,)] * self.grid_len + observation[..., (2,)]

        return np.concatenate([observation[..., (0,)], xy, observation[..., 3:]])