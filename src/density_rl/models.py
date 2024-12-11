from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.distributions import OneHotCategorical, Distribution


def _layer_init(layer, std=1.414, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _dense_init(nb_inputs, nb_outputs, layers, activation):
    # network layers
    net = []

    # build the net up to the last hidden layer
    input_size = nb_inputs
    for output_size in layers:
        net.append(_layer_init(nn.Linear(input_size, output_size)))
        net.append(activation())

        input_size = output_size

    # add a linear output layer
    net.append(_layer_init(nn.Linear(input_size, nb_outputs)))

    return net


class MultiOneHotCategorical(Distribution):
    def __init__(self, vect_logits):
        super().__init__()
        self.categories_len = tuple([v.shape[-1] for v in vect_logits])
        self.vect_one_hot_categorical = [OneHotCategorical(logits=logits) for logits in vect_logits]

    def log_prob(self, x):
        return torch.sum(self.log_prob_joint(x), dim=-1)

    def log_prob_joint(self, x):
        categorical_sa = x.split(self.categories_len, dim=-1)
        return torch.stack([model.log_prob(sa)
                            for sa, model in zip(categorical_sa, self.vect_one_hot_categorical)], dim=-1)

    def sample(self, sample_shape=torch.Size([])):
        return torch.cat([categorical.sample() for categorical in self.vect_one_hot_categorical], dim=-1)


class ForwardModel(nn.Module):
    def __init__(self, condition_size, nb_outputs, layers, activation=nn.Tanh):
        super().__init__()

        # initialize the sequential model
        self._net = nn.Sequential(*_dense_init(condition_size, nb_outputs, layers, activation))

    def forward(self, x):
        return self._net(x)


class CategoricalModel(nn.Module):
    def __init__(self, condition_size, nb_classes, layers, activation=nn.Tanh):
        super().__init__()

        # initialize the sequential model
        self._net = nn.Sequential(*_dense_init(condition_size, nb_classes, layers, activation))

    def forward(self, x):
        return OneHotCategorical(logits=self._net(x))


class MultiCategoricalMarginalModel(nn.Module):
    def __init__(self, vect_classes):
        super().__init__()

        # initialize the sequential model
        self._vect_classes = tuple(vect_classes)
        self._nb_parameters = torch.tensor(self._vect_classes).sum().item()
        self._logits = nn.Parameter(torch.zeros(self._nb_parameters))

    def forward(self, x):
        batch_shape = x.shape[:-1]
        logits_reshaped = self._logits.view((1,) * len(batch_shape) + (self._nb_parameters,))
        logits_repeated = logits_reshaped.expand(*batch_shape, self._nb_parameters)
        vect_logits_repeated = logits_repeated.split(self._vect_classes, dim=-1)
        return MultiOneHotCategorical(vect_logits=vect_logits_repeated)


class MultiCategoricalModel(nn.Module):
    def __init__(self, condition_size, vect_classes, layers, activation=nn.Tanh):
        super().__init__()

        # initialize the sequential model
        self._vect_classes = tuple(vect_classes)
        self._net = nn.Sequential(*_dense_init(condition_size, int(vect_classes.sum()), layers, activation))

    def forward(self, x):
        vect_logits = self._net(x).split(self._vect_classes, dim=-1)
        return MultiOneHotCategorical(vect_logits=vect_logits)


class MultiCategoricalAutoRegressiveModel(nn.Module):
    def __init__(self, conditioning_size,
                 categories_len,
                 input_projection_layers=(),
                 rnn_input_size=256,
                 rnn_hidden_size=128,
                 rn_num_layers=1,
                 dense_net_layers=(128, 128,),
                 output_context_size=256,
                 output_projection_layers=()):
        super().__init__()

        self._conditioning_size = conditioning_size  # size of the context
        self._categories_len = list(categories_len)  # tuple with the number of categories

        # initial hidden state of the RNN
        self._h0 = torch.zeros(rn_num_layers, 1, rnn_hidden_size)

        # networks for computing log_prob and for sampling
        self._projection_rec_logits = [nn.Sequential(*_dense_init(prev_input_size + conditioning_size, rnn_input_size,
                                                                  input_projection_layers, nn.ReLU))
                                       for prev_input_size in [0] + self._categories_len[:-1]]
        self._rnn = torch.nn.GRU(rnn_input_size, rnn_hidden_size, num_layers=rn_num_layers)
        self._seqnn = nn.Sequential(*_dense_init(rn_num_layers * rnn_hidden_size + conditioning_size,
                                                 output_context_size, dense_net_layers, nn.ReLU))
        self._projection_logits = [nn.Sequential(*_dense_init(output_context_size + prev_input_size + conditioning_size,
                                                              output_size, output_projection_layers, nn.ReLU))
                                   for output_size, prev_input_size
                                   in zip(categories_len, [0] + self._categories_len[:-1])]

    def forward(self, x):
        return MultiCategoricalAutoRegressiveModel.MultiOneHotCategorical(
            self._categories_len, x, self._h0, self._projection_rec_logits, self._rnn, self._seqnn,
            self._projection_logits)

    class MultiOneHotCategorical(Distribution):
        def __init__(self, categories_len, context, h0, projection_rec_logits, rnn, seqnn, projection_logits):
            super().__init__()
            self.categories_len = categories_len

            # conditioning context
            self._context = context

            # initial hidden state of the RNN
            self._h0 = h0

            # networks for computing log_prob and for sampling
            self._projection_rec_logits = projection_rec_logits  # (state, context) -> RNN input
            self._rnn = rnn  # (RNN input, h_{n-1}) -> RNN output, RNN h_n
            self._seqnn = seqnn  # (RNN output, context) -> seq nn output
            self._projection_logits = projection_logits  # (seq nn output, state, context) -> logits next state

        def log_prob_joint(self, x):
            x_original_shape = x.shape
            x = torch.flatten(x, 0, -2)
            context = torch.flatten(self._context, 0, -2)

            categorical_sa = x.split(self.categories_len, dim=-1)

            conditional_log_prob = []
            regression_input = context
            hn = self._h0.expand(-1, context.shape[0], -1)

            for sa, in_layer, out_layer in zip(categorical_sa, self._projection_rec_logits, self._projection_logits):
                # compute the logits of the first state give the rnn input
                rnn_input = in_layer(regression_input).unsqueeze(0)
                _, hn = self._rnn(rnn_input, hn)
                rnn_output = torch.flatten(torch.swapaxes(hn, 0, 1), 1, -1)
                seq_nn_output = self._seqnn(torch.cat([rnn_output, context], dim=-1))
                logits = out_layer(torch.cat([seq_nn_output, regression_input], dim=-1))

                # get the logprob
                conditional_log_p = torch.sum(sa * torch.nn.functional.log_softmax(logits), dim=-1, keepdim=True)
                conditional_log_prob.append(conditional_log_p)

                # update the input
                regression_input = torch.cat([sa, context], dim=-1)

            conditional_log_prob = torch.cat(conditional_log_prob, dim=-1)

            return conditional_log_prob.reshape(x_original_shape[:-1] + (-1,))

        def log_prob(self, x):
            return torch.sum(self.log_prob_joint(x), dim=-1)

        def sample(self, sample_shape=torch.Size([])):
            context_original_shape = self._context.shape
            context = torch.flatten(self._context, 0, -2)

            conditional_samples = []
            regression_input = context
            hn = self._h0.expand(-1, context.shape[0], -1)

            for in_layer, out_layer in zip(self._projection_rec_logits, self._projection_logits):
                # compute the logits of the first state give the rnn input
                rnn_input = in_layer(regression_input).unsqueeze(0)
                _, hn = self._rnn(rnn_input, hn)
                rnn_output = torch.flatten(torch.swapaxes(hn, 0, 1), 1, -1)
                seq_nn_output = self._seqnn(torch.cat([rnn_output, context], dim=-1))
                logits = out_layer(torch.cat([seq_nn_output, regression_input], dim=-1))

                # sample a value
                sa_id = torch.squeeze(torch.multinomial(torch.nn.functional.softmax(logits), 1), dim=-1)
                sa = nn.functional.one_hot(sa_id, logits.shape[-1])
                conditional_samples.append(sa)

                # update the input
                regression_input = torch.cat([sa, context], dim=-1)

            conditional_samples = torch.cat(conditional_samples, dim=-1)

            return conditional_samples.reshape(context_original_shape[:-1] + (-1,))


class ActorFeatureVisitationAgent(nn.Module, ABC):
    def __init__(self, actor_model, visitation_model):
        super().__init__()
        self._actor_model = actor_model
        self._visitation_model = visitation_model

    @property
    def actor_model(self):
        return self._actor_model

    @property
    def visitation_model(self):
        return self._visitation_model

    @property
    @abstractmethod
    def feature_map(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_model(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def relative_density(self):
        raise NotImplementedError


class ActorVisitationAgent(ActorFeatureVisitationAgent):
    @property
    def feature_map(self):
        def identity(x):
            return x

        return identity

    @property
    def feature_model(self):
        return self.visitation_model

    @property
    def relative_density(self):
        def uniform(x):
            return torch.zeros_like(x[..., 0])

        return uniform


class ActorCriticVisitationAgent(ActorVisitationAgent):
    def __init__(self, actor_model, critic_model, visitation_model):
        super().__init__(actor_model, visitation_model)
        self._critic_model = critic_model

    @property
    def critic_model(self):
        return self._critic_model
