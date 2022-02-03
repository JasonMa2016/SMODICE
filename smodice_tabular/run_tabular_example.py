from copy import copy 
import os
import time
from collections import defaultdict

import numpy as np

from utils import (SMODICE, generate_trajectory, policy_execution, compute_marginal_distribution,)
from tabular_mdps import ExampleMDP, compute_MLE_TabularMDP

seed = 20
dims = (9,9) 

mdp = ExampleMDP(dims, 4, initial_state=20, absorbing_state=60, gamma=0.99, stochasticity=0.1)

# Create random policy pi_b
pi_b = np.ones((mdp.S, mdp.A)) / mdp.A

# Use pi_b to collect trajectories
trajectory_offline = generate_trajectory(seed, mdp, pi_b, num_episodes=10000)

# Compute empirical MDP
mdp_all, N_all = compute_MLE_TabularMDP(mdp, mdp.dims, mdp.A, mdp.R, mdp.gamma, trajectory_offline)

# Compute SMODICE policy
pi_smodice = SMODICE(mdp_all, mdp, None, pi_b, alpha=1., delta=0.0001, example=True)[0]

# Print the success state and SMODICE policy
path_learner = policy_execution(mdp, pi_smodice)
print("Success State:")
print(mdp.goal_state)
print("SMODICE Policy:")
print(path_learner)

# Print the state occupancies for all states
d_pi = compute_marginal_distribution(mdp, pi_smodice)
d_pi_s = d_pi.reshape(mdp.S, mdp.A).sum(axis=1)
print(np.sort(d_pi_s))
for i, d in enumerate(d_pi_s):
    print(i, d)
print(np.sum(d_pi_s))