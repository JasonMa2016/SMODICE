import numpy as np 
import torch 
from networks_pytorch import TanhMixtureNormalPolicy, TanhNormalPolicy, ValueNetwork, QNetwork


class SMODICE(object):
    def __init__(self, observation_spec, action_spec, config):
        self._disc_type = config['disc_type']
        self._gamma = config['gamma']
        self._env_name = config['env_name']
        self._total_iterations = config['total_iterations']
        self._use_policy_entropy_constraint = config['use_policy_entropy_constraint']
        self._target_entropy = config['target_entropy']
        self._hidden_sizes = config['hidden_sizes']
        self._batch_size = config['batch_size']
        self._f = config['f']
        self._lr = config['lr']
        self._actor_lr = config['actor_lr']
        self._v_l2_reg = config['v_l2_reg']

        self.device = config['device']

        self._iteration = 0
        self._optimizers = dict()

        self._v_network = ValueNetwork(observation_spec, hidden_sizes=self._hidden_sizes).to(self.device)
        self._optimizers['v'] = torch.optim.Adam(self._v_network.parameters(), self._lr, weight_decay=self._v_l2_reg)

        # f-divergence functions
        if self._f == 'chi':
            self._f_fn = lambda x: 0.5 * (x - 1) ** 2
            self._f_star_prime = lambda x: torch.relu(x + 1)
            self._f_star = lambda x: 0.5 * x ** 2 + x 
        elif self._f == 'kl':
            self._f_fn = lambda x: x * torch.log(x + 1e-10)
            self._f_star_prime = lambda x: torch.exp(x - 1)
        else:
            raise NotImplementedError()

        # policy
        self._policy_network = TanhNormalPolicy(observation_spec, action_spec, hidden_sizes=self._hidden_sizes,
                                                mean_range=config['mean_range'], logstd_range=config['logstd_range']).to(self.device)
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)

        if self._use_policy_entropy_constraint:
            self._log_ent_coeff = torch.zeros(1, requires_grad=True, device=self.device)
            self._optimizers['ent_coeff'] = torch.optim.Adam([self._log_ent_coeff], self._lr)

    def v_loss(self, initial_v_values, e_v, result={}):
        # Compute v loss
        v_loss0 = (1 - self._gamma) * initial_v_values

        if self._f == 'kl':
            v_loss1 = torch.log(torch.mean(torch.exp(e_v)))
        else:
            v_loss1 = torch.mean(self._f_star(e_v))

        v_loss = v_loss0 + v_loss1
        v_loss = torch.mean(v_loss)

        result.update({
            'v_loss0': torch.mean(v_loss0),
            'v_loss1': torch.mean(v_loss1),
            'v_loss': v_loss,
        })

        return result

    def policy_loss(self, observation, action, w_e, result={}):
        # Compute policy loss
        (sampled_action, sampled_pretanh_action, sampled_action_log_prob, sampled_pretanh_action_log_prob, pretanh_action_dist), _ \
            = self._policy_network((observation,))
        
        # Entropy is estimated on newly sampled action.
        negative_entropy_loss = torch.mean(sampled_action_log_prob)

        # Weighted BC
        action_log_prob, _ = self._policy_network.log_prob(pretanh_action_dist, action, is_pretanh_action=False)
        if self._disc_type == 'bc':
            policy_loss = - torch.mean(action_log_prob)
        else: 
            policy_loss = - torch.mean(w_e * action_log_prob)

        if self._use_policy_entropy_constraint:
            ent_coeff = torch.exp(self._log_ent_coeff).squeeze(0)
            policy_loss += ent_coeff * negative_entropy_loss

            ent_coeff_loss = - self._log_ent_coeff * (sampled_action_log_prob + self._target_entropy).detach()

            result.update({
                'ent_coeff_loss': torch.mean(ent_coeff_loss),
                'ent_coeff': ent_coeff,
            })

        result.update({
            'w_e': w_e,
            'policy_loss': policy_loss,
            'negative_entropy_loss': negative_entropy_loss
        })

        return result

    def train_step(self, initial_observation, observation, action, reward, next_observation, terminal):

        initial_observation = initial_observation.to(self.device)
        observation = observation.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_observation = next_observation.to(self.device)
        terminal = terminal.unsqueeze(1).to(self.device)

        # Shared network values
        initial_v_values, _ = self._v_network((initial_observation,))
        v_values, _ = self._v_network((observation,))
        next_v_values, _ = self._v_network((next_observation,))

        e_v = reward + (1 - terminal) * self._gamma * next_v_values - v_values

        # compute value function loss (Equation 20 in the paper)
        loss_result = self.v_loss(initial_v_values, e_v, result={})

        # extracting importance weight (Equation 21 in the paper)
        if self._f == 'kl':
            w_e = torch.exp(e_v)
        else:
            w_e = self._f_star_prime(e_v)

        # policy learning (Equation 22 in the paper)
        loss_result = self.policy_loss(observation, action, w_e.detach(), result=loss_result)

        self._optimizers['v'].zero_grad()
        loss_result['v_loss'].backward()
        self._optimizers['v'].step()

        self._optimizers['policy'].zero_grad()
        loss_result['policy_loss'].backward()
        self._optimizers['policy'].step()

        if self._use_policy_entropy_constraint:
            self._optimizers['ent_coeff'].zero_grad()
            loss_result['ent_coeff_loss'].backward()
            self._optimizers['ent_coeff'].step()

        self._iteration += 1

        return loss_result

    def step(self, observation):
        """
        observation: batch_size x obs_dim
        """
        observation = torch.from_numpy(observation).to(self.device)
        action = self._policy_network.deterministic_action(observation)

        return action.detach().cpu(), None