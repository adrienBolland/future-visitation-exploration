import torch


class FutureVisitationReward:
    def __init__(self, agent, gamma_state_dist):
        self.agent = agent
        self.gamma_state_dist = gamma_state_dist

    def visitation_loss(self, sa, agent_target=None):
        # get the first elements of the transitions
        obs = sa.observations[:, 0, :]
        actions = sa.actions[:, 0, :]

        # get the bootstrap horizon
        bootstrap_horizon = sa.observations.shape[1]

        # sample the bootstrap
        infinity_bootstrap_batch = future_discounted_batch(sa.observations,
                                                           sa.next_observations,
                                                           sa.dones,
                                                           agent_target if agent_target else self.agent,
                                                           gamma=self.gamma_state_dist,
                                                           bootstrap_n_max=bootstrap_horizon,
                                                           state_action_conditioning=True)

        # get the log likelihood of the current state
        visitation_condition = torch.cat([obs, actions], dim=-1)
        log_prob_s = self.agent.visitation_model(visitation_condition).log_prob(infinity_bootstrap_batch)
        return -torch.mean(log_prob_s)

    def intrinsic_reward(self, sa):
        # get the first elements of the transitions
        obs = sa.observations[:, 0, :]
        actions = sa.actions[:, 0, :]

        # get the bootstrap horizon
        bootstrap_horizon = sa.observations.shape[1]

        infinity_bootstrap_batch = future_discounted_batch(sa.observations,
                                                           sa.next_observations,
                                                           sa.dones,
                                                           self.agent,
                                                           gamma=self.gamma_state_dist,
                                                           bootstrap_n_max=bootstrap_horizon,
                                                           state_action_conditioning=True)

        # get the log likelihood of the current state
        visitation_condition = torch.cat([obs, actions], dim=-1)

        log_prob_feature = self.agent.feature_model(visitation_condition).log_prob(infinity_bootstrap_batch)
        log_prob_relative = self.agent.relative_density(self.agent.feature_map(infinity_bootstrap_batch))
        delta_log_prob = log_prob_relative - log_prob_feature

        return delta_log_prob


def future_discounted_batch(state_seq_batch, next_seq_state_batch, terminated_seq_batch, agent,
                            gamma, bootstrap_n_max, state_action_conditioning):
    # get some values for the problem
    batch_size = state_seq_batch.shape[0]
    nb_transitions = state_seq_batch.shape[1]
    bootstrap_n_max = min(bootstrap_n_max, nb_transitions)

    # compute the next state selection by sampling from a clipped-geometric (! indices starts at 1)
    mode = torch.empty(batch_size).geometric_(1 - gamma)
    mode_clip = torch.clamp(mode, max=bootstrap_n_max)

    # get the next state corresponding to the sampled index
    state_conditioning = next_seq_state_batch[torch.arange(batch_size), (mode_clip - 1).long(), :]

    # sample a bootstrap conditioned on s or (s, a)
    if state_action_conditioning:
        action_batch = agent.actor_model(state_conditioning).sample()
        model_condition = torch.cat([state_conditioning, action_batch], dim=-1)
    else:
        model_condition = state_conditioning

    model_sample = agent.visitation_model(model_condition).sample()

    # compute the mixture weight
    is_not_bootstrapped = (mode == mode_clip).view(-1, 1).float()
    is_bootstrapped = 1 - is_not_bootstrapped

    # compute the first terminal state
    terminated_seq_batch = terminated_seq_batch.squeeze()
    first_terminal = torch.argmax(terminated_seq_batch
                                  + torch.flip(torch.arange(nb_transitions), (0,)) / nb_transitions, dim=-1)
    is_terminal = ((first_terminal >= mode_clip).float()
                   * terminated_seq_batch[torch.arange(batch_size), first_terminal]).view(-1, 1)
    is_not_terminal = 1 - is_terminal

    # terminal states
    terminal = next_seq_state_batch[torch.arange(batch_size), first_terminal, :]

    # weighting of the different cases
    infinite_sample = (is_not_bootstrapped * (is_terminal * terminal + is_not_terminal * state_conditioning)
                       + is_bootstrapped * model_sample)

    return infinite_sample
