import torch


class MarginalVisitationReward:
    def __init__(self, agent, gamma_state_dist):
        self.agent = agent
        self.gamma_state_dist = gamma_state_dist

    def visitation_loss(self, sa):
        obs = sa.observations[:, 0, :]
        prob_obs = (1 - self.gamma_state_dist) * self.gamma_state_dist**sa.time_steps[:, 0]

        visitation_condition = torch.zeros_like(obs)  # only necessary for having the correct shape
        log_prob_s = self.agent.visitation_model(visitation_condition).log_prob(obs)
        return -torch.mean(prob_obs * log_prob_s) / torch.sum(prob_obs)

    def intrinsic_reward(self, sa):
        obs = sa.observations[:, 0, :]

        visitation_condition = torch.zeros_like(obs)  # only necessary for having the correct shape
        log_prob_feature = self.agent.feature_model(visitation_condition).log_prob(obs)
        log_prob_relative = self.agent.relative_density(self.agent.feature_map(obs))
        delta_log_prob = log_prob_relative - log_prob_feature

        return delta_log_prob
