
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Inverse tanh torch function
def atanh(z):
    return 0.5 * (torch.log(1 + z) - torch.log(1 - z))

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_sizes):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear3 = nn.Linear(hidden_sizes[1], 1)

        self.apply(weights_init_)

    def forward(self, state):
        state = torch.cat(state, 1)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x, None

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_sizes=(256, 256)):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear3 = nn.Linear(hidden_sizes[1], 1)

        # # Q2 architecture
        # self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        # self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, inputs):
        xu = torch.cat(inputs, 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # x2 = F.relu(self.linear4(xu))
        # x2 = F.relu(self.linear5(x2))
        # x2 = self.linear6(x2)

        return x1, None

class InverseModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.affine_layers = nn.ModuleList()
        self.layers = 6
        self.first_layer = nn.Linear(self.in_features, hidden_dim)
        for i in range(self.layers):
            self.affine_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()

        self.final = nn.Linear(hidden_dim, out_features)
        self.apply(weights_init_)


    def forward(self, state, next_state):
        inputs = torch.cat((state, next_state), -1)
        last_output = self.relu(self.first_layer(inputs))
        for i, affine in enumerate(self.affine_layers):
            res = self.relu(affine(last_output))
            output = self.relu(last_output+res)
            last_output = output
        action = self.final(last_output)
        return action

    def train(self, states, next_states, actions, optimizer, batch_size=256):
        idxs = np.arange(states.shape[0])
        np.random.shuffle(idxs)
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

        idxs = np.arange(states.shape[0])
        np.random.shuffle(idxs)
        for batch_num in range(num_batch):
            batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
            states_train = states[batch_idxs].float()
            next_states_train = next_states[batch_idxs].float()
            actions_targ = actions[batch_idxs].float()

            res = self.forward(states_train, next_states_train)
            train_losses = ((res - actions_targ) ** 2).mean()
            optimizer.zero_grad()
            train_losses.backward()
            optimizer.step()

    def evaluate(self, states, next_states, actions):
        actions_pred = self.forward(states, next_states)
        mse = ((actions_pred - actions) ** 2).mean(-1).mean(-1)
        return mse.item()

class TanhNormalPolicy(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_sizes=(256,256), action_space=None,
                 mean_range=(-7.24, 7.24), logstd_range=(-5., 2.), eps=1e-6):
        super(TanhNormalPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        self.mean_linear = nn.Linear(hidden_sizes[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[1], num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        
        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def forward(self, inputs, step_type=(), network_state=(), training=False):
        inputs = torch.cat(inputs, 1)
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        logstd = self.log_std_linear(x)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        std = torch.exp(logstd)
        pretanh_action_dist = Normal(mean, std)
        pretanh_action = pretanh_action_dist.rsample()
        action = torch.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.log_prob(pretanh_action_dist, pretanh_action, is_pretanh_action=True)

        return (action, pretanh_action, log_prob, pretanh_log_prob, pretanh_action_dist), network_state

    def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = torch.tanh(pretanh_action)
        else:
            pretanh_action = atanh(torch.clamp(action, -1 + self.eps, 1 - self.eps))

        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
        log_prob = pretanh_log_prob - torch.log(1 - action ** 2 + self.eps)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob, pretanh_log_prob

    def deterministic_action(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        action = torch.tanh(mean)
        return action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(TanhNormalPolicy, self).to(device)


class TanhMixtureNormalPolicy(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_sizes, num_components=2,
                 mean_range=(-9., 9.), logstd_range=(-5., 2.), eps=1e-6, mdn_temperature=1.0):
        super(TanhMixtureNormalPolicy, self).__init__()

        self._num_components = num_components 
        self._mdn_temperature = mdn_temperature

        self.linear1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.mean_linear = nn.Linear(hidden_sizes[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[1], num_actions)
        self.logits_linear = nn.Linear(hidden_sizes[1], num_actions)

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def forward(self, inputs, step_type=(), network_state=(), training=False):
        inputs = torch.cat(inputs, 1)
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))

        means = self.mean_linear(x)
        means = torch.clamp(means, self.mean_min, self.mean_max)
        means = torch.reshape(means, (-1, self._num_components, self._action_dim))
        logstds = self.log_std_linear(x)
        logstds = torch.clamp(logstds, self.logstd_min, self.logstd_max)
        logstds = torch.reshape(logstds, (-1, self._num_components, self._action_dim))
        stds = torch.exp(logstds)

        component_logits = self.logits_linear(x) / self._mdn_temperature

        pretanh_actions_dist = Normal(means, stds)
        component_dist = Categorical(logits=component_logits)

        pretanh_actions = pretanh_actions_dist.rsample()  # (batch_size, num_components, action_dim)
        component = component_dist.rsample()  # (batch_size)

        batch_idx = torch.range(torch.shape(inputs[0])[0])
        pretanh_action = torch.gather_nd(pretanh_actions, torch.stack([batch_idx, component], axis=1))
        action = torch.tanh(pretanh_action)

        log_prob, pretanh_log_prob = self.log_prob((component_dist, pretanh_actions_dist), pretanh_action, is_pretanh_action=True)

        return (action, pretanh_action, log_prob, pretanh_log_prob, (component_dist, pretanh_actions_dist)), network_state

    def log_prob(self, dists, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = torch.tanh(pretanh_action)
        else:
            pretanh_action = atanh(torch.clamp(action, -1 + self.eps, 1 - self.eps))

        component_dist, pretanh_actions_dist = dists
        component_logits = component_dist.logits_parameter()
        component_log_prob = component_logits - torch.math.reduce_logsumexp(component_logits, axis=-1, keepdims=True)

        pretanh_actions = torch.tile(pretanh_action[:, None, :], (1, self._num_components, 1))  # (batch_size, num_components, action_dim)

        pretanh_log_prob = torch.reduce_logsumexp(component_log_prob + pretanh_actions_dist.log_prob(pretanh_actions), axis=1)
        log_prob = pretanh_log_prob - torch.math.log(1 - action ** 2 + self.eps)
        log_prob = log_prob.sum(1, keepdim=True)

        return log_prob, pretanh_log_prob
