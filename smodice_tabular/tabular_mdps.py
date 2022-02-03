import numbers
import os
from copy import copy

import numpy as np
from utils import solve_MDP 

def gridworld_P(dims, stochasticity=0., speed=1):
    assert len(dims) == 2
    h,w = dims
    state_space_size = dims[0] * dims[1]
    action_space_size = 4
    UP, DOWN, LEFT, RIGHT = range(action_space_size)
    P = np.zeros([state_space_size, action_space_size, state_space_size])
    for state_i in range(state_space_size):
        x,y = state_i % w, state_i // w
        # locate adjactent idxs
        up_i = ((y-speed)%h)*w + (x)%w
        down_i = ((y+speed)%h)*w + (x)%w
        left_i = ((y)%h)*w + (x-speed)%w
        right_i = ((y)%h)*w + (x+speed)%w
        # intentional action probabilities
        P[state_i, UP, up_i] = 1. - stochasticity
        P[state_i, DOWN, down_i] = 1. - stochasticity
        P[state_i, LEFT, left_i] = 1. - stochasticity
        P[state_i, RIGHT, right_i] = 1. - stochasticity
        # sometimes the agent moves in a random direction instead
        for i in [up_i, down_i, left_i, right_i]: P[state_i, :, i] += stochasticity * (1 / action_space_size)
    return P

def diagonal_gridworld_P(dims, stochasticity=0., speed=1):
    assert len(dims) == 2
    h,w = dims
    state_space_size = dims[0] * dims[1]
    action_space_size = 4
    UP_LEFT, DOWN_LEFT, UP_RIGHT, DOWN_RIGHT = range(action_space_size)
    P = np.zeros([state_space_size, action_space_size, state_space_size])
    for state_i in range(state_space_size):
        x,y = state_i % w, state_i // w
        # locate adjactent idxs
        up_left_i = ((y-speed)%h)*w + (x-speed)%w
        down_left_i = ((y+speed)%h)*w + (x-speed)%w
        up_right_i = ((y-speed)%h)*w + (x+speed)%w
        down_right_i = ((y+speed)%h)*w + (x+speed)%w
        # intentional action probabilities
        P[state_i, UP_LEFT, up_left_i] = 1. - stochasticity
        P[state_i, DOWN_LEFT, down_left_i] = 1. - stochasticity
        P[state_i, UP_RIGHT, up_right_i] = 1. - stochasticity
        P[state_i, DOWN_RIGHT, down_right_i] = 1. - stochasticity
        # sometimes the agent moves in a random direction instead
        for i in [up_left_i, down_left_i, up_right_i, down_right_i]: P[state_i, :, i] += stochasticity * (1 / action_space_size)
    return P


class DomainAdaptiveMDP:

    def __init__(self, dims=(8,8), A=4, initial_state=0, absorbing_state=25, T=None, gamma=0.95, stochasticity=0., diagonal=False):
        """
        Create a random MDP

        :param S: the number of states
        :param A: the number of actions
        :param gamma: discount factor
        """
        self.gamma = gamma
        self.dims = dims 
        self.S = dims[0] * dims[1]
        self.A = A
        self.initial_state = initial_state
        self.absorbing_state = absorbing_state
        self.stochasticity = stochasticity 

        self.diagonal = diagonal 
        if T is None:
            if not self.diagonal:
                self.T = gridworld_P(dims, stochasticity=stochasticity)
            else:
                self.T = diagonal_gridworld_P(dims, stochasticity=stochasticity)
            self.T[self.absorbing_state, :, :] = 0
            self.T[self.absorbing_state, :, self.absorbing_state] = 1
        else:
            self.T = np.array(T)
        self.R = np.zeros((self.S, self.A))
        self.goal_state = self.absorbing_state
        self.R[self.goal_state, : ] = 1.

    def change_speed(self, speed=1):
        if not self.diagonal:
            self.T = gridworld_P(self.dims, stochasticity=self.stochasticity, speed=speed)
        else:
            self.T = diagonal_gridworld_P(self.dims, stochasticity=self.stochasticity, speed=speed)
        self.T[self.absorbing_state, :, :] = 0
        self.T[self.absorbing_state, :, self.absorbing_state] = 1

    def __copy__(self):
        mdp = DomainAdaptiveMDP(dims=self.dims, A=self.A, initial_state=self.initial_state,
        absorbing_state=self.absorbing_state, stochasticity=self.stochasticity,
        T=self.T, gamma=self.gamma)
        return mdp

class ExampleMDP:

    def __init__(self, dims=(8,8), A=4, initial_state=0, absorbing_state=25, T=None, gamma=0.95, stochasticity=0.2):
        """
        Create a random MDP

        :param S: the number of states
        :param A: the number of actions
        :param gamma: discount factor
        """
        self.gamma = gamma
        self.dims = dims 
        self.S = dims[0] * dims[1]
        self.A = A
        self.initial_state = initial_state
        self.absorbing_state = absorbing_state
        self.stochasticity = stochasticity 

        if T is None:
            self.T = gridworld_P(dims, stochasticity=stochasticity)
            self.T[self.absorbing_state, :, :] = 0
            self.T[self.absorbing_state, :, self.absorbing_state] = 1
        else:
            self.T = np.array(T)
        self.R = np.zeros((self.S, self.A))
        self.goal_state = self.absorbing_state
        self.R[self.goal_state, : ] = 1.

    def __copy__(self):
        mdp = ExampleMDP(dims=self.dims, A=self.A, initial_state=self.initial_state,
        absorbing_state=self.absorbing_state, stochasticity=self.stochasticity,
        T=self.T, gamma=self.gamma)
        return mdp

class ExampleMDP2:

    def __init__(self, dims=(8,8), A=4, initial_state=0, absorbing_states=[25], T=None, gamma=0.95, stochasticity=0.2):
        """
        Create a random MDP

        :param S: the number of states
        :param A: the number of actions
        :param gamma: discount factor
        """
        self.gamma = gamma
        self.dims = dims 
        self.S = dims[0] * dims[1]
        self.A = A
        self.initial_state = initial_state
        self.absorbing_states = absorbing_states
        self.stochasticity = stochasticity 

        if T is None:
            self.T = gridworld_P(dims, stochasticity=stochasticity)
            for state in self.absorbing_states:
                self.T[state, :, :] = 0
                self.T[state, :, state] = 1
        else:
            self.T = np.array(T)
        self.R = np.zeros((self.S, self.A))
        # self.goal_state = self.absorbing_state
        for state in self.absorbing_states:
            self.R[state, : ] = 1.

    def __copy__(self):
        mdp = ExampleMDP2(dims=self.dims, A=self.A, initial_state=self.initial_state,
        absorbing_states=self.absorbing_states, stochasticity=self.stochasticity,
        T=self.T, gamma=self.gamma)
        return mdp

def compute_MLE_TabularMDP(mdp, dims, A, R, gamma, trajectory, absorb_unseen=True):
    S = mdp.S
    N = np.zeros((S, A, S))
    for trajectory_one in trajectory:
        for episode, t, state, action, reward, state1 in trajectory_one:
            N[state, action, state1] += 1

    T = gridworld_P(dims, stochasticity=mdp.stochasticity)
    T[mdp.absorbing_state, :, :] = 0
    T[mdp.absorbing_state, :, mdp.absorbing_state] = 1
    for s in range(S):
        for a in range(A):
            if N[s, a, :].sum() != 0 and s!=mdp.absorbing_state:
                T[s, a, :] = N[s, a, :] / N[s, a, :].sum()
    mle_mdp = copy(mdp)
    mle_mdp.T = T

    return mle_mdp, N