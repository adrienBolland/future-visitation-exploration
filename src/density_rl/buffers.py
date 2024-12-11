import torch
import typing

import numpy as np


class TransitionsReplayBufferSamples(typing.NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    time_steps: torch.Tensor


class TransitionsReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_size: int,
        action_size: int,
        transition_size: int,
        device: typing.Union[torch.device, str] = "auto",
    ):
        self.buffer_size = buffer_size
        self.observation_size = observation_size
        self.action_size = action_size
        self.transition_size = transition_size
        self.pos = 0
        self.full = False

        self.device = device

        self.observations = np.zeros((self.buffer_size, self.observation_size), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.observation_size), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_size), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.time_steps = np.zeros((self.buffer_size, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: typing.List[typing.Dict[str, typing.Any]],
        time_step: int
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)
        self.next_observations[self.pos] = np.array(next_obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.time_steps[self.pos] = np.array(time_step)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> TransitionsReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        batch_inds_trans = (batch_inds[:, np.newaxis] + np.arange(self.transition_size)) % self.buffer_size
        batch_inds_trans = np.reshape(batch_inds_trans, (-1,))

        data = (np.reshape(arr[batch_inds_trans], (batch_size, self.transition_size, -1))
                for arr in (self.observations, self.actions, self.next_observations, self.dones, self.rewards, self.time_steps))

        return TransitionsReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
