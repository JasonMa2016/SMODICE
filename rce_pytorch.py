import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = torch.sigmoid(self.l3(q1))

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = torch.sigmoid(self.l6(q2))
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = torch.sigmoid(self.l3(q1))
        return q1


class RCE_TD3_BC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.gamma = self.discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.device = device 
        self.total_it = 0


    def step(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).detach().cpu(), None

    def train_step(self, success_state, state, action, reward, next_state, terminal):
        self.total_it += 1
        success_state = success_state.to(self.device)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        not_done = 1 - terminal.unsqueeze(1).to(self.device)

        batch_size = state.shape[0]

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

        # Equation (9) in the RCE paper
        # prevents NaN
        w = target_Q / (1 - target_Q + 1e-5)
       
        # targets in the Cross-Entropy losses
        td_targets = self.gamma * w / (self.gamma * w + 1)
        td_targets = td_targets.detach()
        if not (torch.all(td_targets >=0.) and torch.all(td_targets <= 1.)):
            from ipdb import set_trace
            set_trace() 

        assert(torch.all(td_targets >=0.) and torch.all(td_targets <= 1.))
        td_targets = torch.cat([td_targets, torch.ones(batch_size).unsqueeze(1).to(td_targets.device)])
        
        # Weighs the two Cross-Entropy losses
        weights = torch.cat([1 + self.gamma * w, (1 - self.gamma) * torch.ones(batch_size).unsqueeze(1).to(td_targets.device)])
        
        # Get current Q estimates for success states and offline data
        with torch.no_grad():
            expert_action = self.actor_target(success_state)
        state_combined = torch.cat([state, success_state], axis=0)
        action_combined = torch.cat([action, expert_action], axis=0)        
        current_Q1, current_Q2 = self.critic(state_combined, action_combined)
        assert(torch.all(current_Q1 >=0.) and torch.all(current_Q1 <= 1.))
        assert(torch.all(current_Q2 >=0.) and torch.all(current_Q2 <= 1.))

        # Compute critic loss, Equation (10) in the RCE paper
        critic_loss = F.binary_cross_entropy(current_Q1, td_targets, weight=weights) + F.binary_cross_entropy(current_Q2, td_targets, weight=weights)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        train_results = {'critic_loss': critic_loss}

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha/Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action) 
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            train_results.update({'actor_loss': actor_loss})
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return train_results

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)