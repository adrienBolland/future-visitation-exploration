import torch

from .models import ActorFeatureVisitationAgent


class MinigridPositionModel:

    def __init__(self, visitation_model, xy_model=False):
        self._visitation_model = visitation_model
        self._xy_model = xy_model

    def __call__(self, conditioning):
        return MinigridPositionModel.PositionDist(conditioning, self._visitation_model, self._xy_model)

    class PositionDist:
        def __init__(self, conditioning, visitation_model, xy_model):
            self._visitation_dist = visitation_model(conditioning)
            self._xy_model = xy_model

        def log_prob(self, x):
            joint_dist = self._visitation_dist.log_prob_joint(x)
            if self._xy_model:
                feature_log_prob = joint_dist[..., 1]
            else:
                feature_log_prob = joint_dist[..., 1] + joint_dist[..., 2]

            return feature_log_prob

        def sample(self):
            raise NotImplementedError


class MinigridPositionVisitationAgent(ActorFeatureVisitationAgent):

    def __init__(self, actor_model, visitation_model, env, xy_model=False):
        super().__init__(actor_model, visitation_model)

        # if true, there a single state for the xy positions, otherwise there are two
        self._xy_model = xy_model

        self._observation_space = tuple(env.observation_space.nvec)
        self._id_0 = self._observation_space[0]

        if self._xy_model:
            self._id_1 = self._observation_space[0] + self._observation_space[1]
        else:
            self._id_1 = self._observation_space[0] + self._observation_space[1] + self._observation_space[2]

    @property
    def feature_map(self):
        def get_position(state):
            return state[..., self._id_0:self._id_1]

        return get_position

    @property
    def feature_model(self):
        return MinigridPositionModel(self.visitation_model, self._xy_model)

    @property
    def relative_density(self):
        def uniform(x):
            return torch.zeros_like(x[..., 0])

        return uniform


class MinigridPositionActorCriticVisitationAgent(MinigridPositionVisitationAgent):
    def __init__(self, actor_model, critic_model, visitation_model, env, xy_model=False):
        super().__init__(actor_model, visitation_model, env, xy_model)
        self._critic_model = critic_model

    @property
    def critic_model(self):
        return self._critic_model