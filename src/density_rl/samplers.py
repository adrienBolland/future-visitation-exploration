import torch


class TransitionSamplerIt:
    def __init__(self, env, agent):
        self._env = env
        self._agent = agent
        self._obs = None
        self.reset_env()

    def sample(self, nb_transitions):

        for _ in range(nb_transitions):
            action = self._agent.actor_model(torch.tensor(self._obs, dtype=torch.float)).sample().numpy()
            next_obs, reward, terminated, truncated, info = self._env.step(action)

            yield self._obs, action, reward, next_obs, terminated, truncated, info

            if terminated or truncated:
                self.reset_env()
            else:
                self._obs = next_obs

    def reset_env(self):
        self._obs, _ = self._env.reset()


class TransitionSampler:
    def __init__(self, env, agent):
        self._env = env
        self._agent = agent
        self._obs = None
        self.reset_env()

    def sample(self, nb_transitions):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []

        for _ in range(nb_transitions):
            action = self._agent.actor_model(self._obs).sample()
            next_obs, reward, terminated, truncated, info = self._env.step(action.numpy())

            next_obs = torch.tensor(next_obs, dtype=torch.float)
            reward = torch.tensor(reward, dtype=torch.float)

            state_list.append(self._obs)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_obs)

            if terminated or truncated:
                self.reset_env()
            else:
                self._obs = next_obs

        return [torch.stack(stacked_list, dim=0)
                for stacked_list in [state_list, action_list, reward_list, next_state_list]]

    def reset_env(self):
        obs, _ = self._env.reset()
        self._obs = torch.tensor(obs, dtype=torch.float)


class TrajectorySampler:
    def __init__(self, env, agent):
        self._env = env
        self._agent = agent
        self._sampler = TransitionSampler(self._env, self._agent )

    def sample(self, nb_trajectories, length_trajectory):
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []

        for _ in range(nb_trajectories):
            self._sampler.reset_env()
            for sequence, container in zip(self._sampler.sample(length_trajectory),
                                           [states_list, actions_list, rewards_list, next_states_list]):
                container.append(sequence)

        return [torch.stack(stacked_list, dim=0)
                for stacked_list in [states_list, actions_list, rewards_list, next_states_list]]

    def reset_env(self):
        self._sampler.reset_env()