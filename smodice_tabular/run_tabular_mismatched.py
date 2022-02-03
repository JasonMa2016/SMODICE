from copy import copy 
import os
import time
from collections import defaultdict

import numpy as np

from utils import (SMODICE, generate_trajectory, policy_execution, solve_MDP)
from tabular_mdps import DomainAdaptiveMDP, compute_MLE_TabularMDP

seed = 20
dims = (9,9) 

mdp = DomainAdaptiveMDP(dims, 4, initial_state=20, absorbing_state=60, gamma=0.99, stochasticity=0.1)

# Create expert-MDP and solve for pi_star (the expert policy)
mdp_expert = DomainAdaptiveMDP(dims, 4, initial_state=20, absorbing_state=60, gamma=0.99, stochasticity=0.1, diagonal=True)
pi_star, _, _ = solve_MDP(mdp_expert)

# Create random policy pi_b
pi_b = np.ones((mdp.S, mdp.A)) / mdp.A

# Use pi_b to collect trajectories
trajectory_offline = generate_trajectory(seed, mdp, pi_b, num_episodes=10000)

# Compute empirical MDP
mdp_all, N_all = compute_MLE_TabularMDP(mdp, mdp.dims, mdp.A, mdp.R, mdp.gamma, trajectory_offline)

# Compute SMODICE policy
pi_smodice = SMODICE(mdp_all, mdp_expert, pi_star, pi_b, alpha=1., delta=0.0001)[0]

# Print out expert vs. SMODICE policies
path_expert = policy_execution(mdp_expert, pi_star)
path_learner = policy_execution(mdp, pi_smodice)

print("Expert Policy:")
print(path_expert)
print("SMODICE Policy:")
print(path_learner)

