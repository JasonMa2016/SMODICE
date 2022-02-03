import d4rl 
import gym 
import numpy as np
import pickle 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from tqdm import tqdm 

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/gail.py
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, device='cuda:0'):
        super(Discriminator, self).__init__()

        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=3e-4)

    def compute_grad_pen(self,
                         expert_state,
                         offline_state,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = expert_state 
        offline_data = offline_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * offline_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, offline_loader):
        self.train()

        loss = 0
        n = 0
        for expert_state, offline_state in zip(expert_loader, offline_loader):

            expert_state = expert_state[0].to(self.device)
            offline_state = offline_state[0][:expert_state.shape[0]].to(self.device)

            policy_d = self.trunk(offline_state)
            expert_d = self.trunk(expert_state)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, offline_state)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state):
        with torch.no_grad():
            self.eval()
            d = self.trunk(state)
            s = torch.sigmoid(d)
            # log(d^E/d^O)
            # reward  = - (1/s-1).log()
            reward = s.log() - (1 - s).log()
            return reward 


class Discriminator_SA(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cuda:0'):
        super(Discriminator_SA, self).__init__()

        self.device = device
        self.state_dim = state_dim 
        self.action_dim = action_dim
        state_hidden_dim = hidden_dim if action_dim == 0 else int(hidden_dim/2) 
        self.state_trunk = nn.Sequential(
            nn.Linear(state_dim, state_hidden_dim), nn.Tanh()).to(device)
        action_trunk_input_dim = 1 if action_dim == 0 else action_dim 
        self.action_trunk = nn.Sequential(
            nn.Linear(action_trunk_input_dim, int(hidden_dim/2)), nn.Tanh()).to(device)
        
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.state_trunk.train()
        self.action_trunk.train()
        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=3e-4)

    def forward(self, input):
        if input.shape[1] == self.state_dim:
            h = self.state_trunk(input)
            h = self.trunk(h)
        else:
            state = input[:, :self.state_dim]
            action = input[:, self.state_dim:]
            h_state = self.state_trunk(state)
            h_action = self.action_trunk(action)
            h = torch.cat([h_state, h_action], axis=1)
            h = self.trunk(h)
        return h 
        
    def compute_grad_pen(self,
                         expert_state,
                         offline_state,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = expert_state 
        offline_data = offline_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * offline_data
        mixup_data.requires_grad = True

        disc = self(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, offline_loader):
        self.train()

        loss = 0
        n = 0
        for expert_state, offline_state in zip(expert_loader, offline_loader):

            expert_state = expert_state[0].to(self.device)
            offline_state = offline_state[0][:expert_state.shape[0]].to(self.device)

            policy_d = self(offline_state)
            expert_d = self(expert_state)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, offline_state)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def update2(self, expert_loader, offline_loader):
        self.train()

        loss = 0
        n = 0
        for offline_state in tqdm(offline_loader):
            for expert_state in expert_loader:
                batch_size = min(offline_state[0].shape[0], expert_state[0].shape[0])
                offline_state = offline_state[0][:batch_size].to(self.device)
                expert_state = expert_state[0][:batch_size].to(self.device)
                policy_d = self(offline_state)
                expert_d = self(expert_state)

                expert_loss = F.binary_cross_entropy_with_logits(
                    expert_d,
                    torch.ones(expert_d.size()).to(self.device))
                policy_loss = F.binary_cross_entropy_with_logits(
                    policy_d,
                    torch.zeros(policy_d.size()).to(self.device))

                gail_loss = expert_loss + policy_loss
                grad_pen = self.compute_grad_pen(expert_state, offline_state)

                loss += (gail_loss + grad_pen).item()
                n += 1

                self.optimizer.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer.step()
                # force just once in the inner-loop 
                break  
        return loss / n

    def predict_reward(self, state):
        with torch.no_grad():
            self.eval()
            d = self(state)
            s = torch.sigmoid(d)
            # log(d^E/d^O)
            # reward  = - (1/s-1).log()
            reward = s.log() - (1 - s).log()
            return reward 
