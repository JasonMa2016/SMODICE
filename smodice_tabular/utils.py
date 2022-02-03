import numbers
import os
from copy import copy

import numpy as np


class MDP:

    def __init__(self, S=50, A=4, T=None, R=None, gamma=0.95):
        """
        Create a random MDP

        :param S: the number of states
        :param A: the number of actions
        :param gamma: discount factor
        """
        self.gamma = gamma
        self.S = S
        self.A = A
        self.initial_state = 0
        self.absorbing_state = S - 1

        if T is None:
            self.T = np.zeros((self.S, self.A, self.S))
            for s in range(S):
                if s == self.absorbing_state:
                    self.T[s, :, s] = 1  # absorbing state: self-transition
                else:
                    for a in range(A):
                        p = np.r_[np.random.dirichlet([1, 1, 1, 1]), [0] * (S - 4 - 1)]
                        np.random.shuffle(p)
                        self.T[s, a, :] = np.r_[p, [0]]
        else:
            self.T = np.array(T)

        if R is None:
            min_value_state, min_value = -1, 1e10
            for s in range(S - 1):
                self.R = np.zeros((self.S, self.A))
                self.R[s, :] = 1
                T_tmp = np.array(self.T[s, :, :])
                self.T[s, :, :] = 0
                self.T[s, :, self.absorbing_state] = 1  # goal_state -> absorbing state
                _, V, _ = solve_MDP(self)
                if V[0] < min_value:
                    min_value = V[0]
                    min_value_state = s
                self.T[s, :, :] = T_tmp

            # Now, we determine the goal state: min_value_state
            self.goal_state = min_value_state
            self.R = np.zeros((self.S, self.A))
            self.R[self.goal_state, :] = 1
            self.T[self.goal_state, :, :] = 0
            self.T[self.goal_state, :, self.absorbing_state] = 1  # goal_state -> absorbing state
        else:
            self.R = np.array(R)

    def __copy__(self):
        mdp = MDP(S=self.S, A=self.A, T=self.T, R=self.R, gamma=self.gamma)
        return mdp


def policy_evaluation(mdp, pi):
    r = np.sum(mdp.R * pi, axis=-1)
    P = np.sum(pi[:, :, None] * mdp.T, axis=1)

    if len(mdp.R.shape) == 3:
        V = np.tensordot(np.linalg.inv(np.eye(mdp.S) - mdp.gamma * P), r, axes=[-1, -1]).T
        Q = mdp.R + mdp.gamma * np.tensordot(mdp.T, V, axes=[-1, -1]).transpose([2, 0, 1])
    else:
        V = np.linalg.inv(np.eye(mdp.S) - mdp.gamma * P).dot(r)
        Q = mdp.R + mdp.gamma * mdp.T.dot(V)
    return V, Q

def policy_execution(mdp, pi):
    path = [mdp.initial_state]
    while len(path) < 50:
        action = np.argmax(pi[path[-1]])
        next_state = np.argmax(mdp.T[path[-1]][action])
        # action = np.random.choice(mdp.A, p=pi[path[-1]])
        # next_state = np.random.choice(mdp.S, p=mdp.T[path[-1]][action])
        path.append(next_state)
        if next_state == mdp.absorbing_state or next_state == mdp.goal_state:
            break
    return path

def solve_MDP(mdp, method='PI'):
    if method == 'PI':
        # policy iteration
        pi = np.ones((mdp.S, mdp.A)) / mdp.A
        V_old = np.zeros(mdp.S)

        for _ in range(1000000):
            V, Q = policy_evaluation(mdp, pi)
            pi_new = np.zeros((mdp.S, mdp.A))
            pi_new[np.arange(mdp.S), np.argmax(Q, axis=1)] = 1.

            if np.all(pi == pi_new) or np.max(np.abs(V - V_old)) < 1e-8:
                break
            V_old = V
            pi = pi_new

        return pi, V, Q
    elif method == 'VI':
        # perform value iteration
        V, Q = np.zeros(mdp.S), np.zeros((mdp.S, mdp.A))
        for _ in range(100000):
            Q_new = mdp.R + mdp.gamma * mdp.T.dot(V)
            V_new = np.max(Q_new, axis=1)

            if np.max(np.abs(V - V_new)) < 1e-8:
                break

            V, Q = V_new, Q_new

        pi = np.zeros((mdp.S, mdp.A))
        pi[np.arange(mdp.S), np.argmax(Q, axis=1)] = 1.

        return pi, V, Q
    else:
        raise NotImplementedError('Undefined method: %s' % method)


def generate_random_mdp(seed, S=50, A=4, gamma=0.95):
    np.random.seed(seed + 1)
    mdp = MDP(S, A, gamma=gamma)
    return mdp


def generate_trajectory(seed, mdp, pi, num_episodes=10, max_timesteps=50, vectorize=True):
    if seed is not None:
        np.random.seed(seed + 1)
    if vectorize:
        def random_choice_prob_vectorized(p):
            """
            e.g. p = np.array([
                [0.1, 0.5, 0.4],
                [0.8, 0.1, 0.1]])
            """
            r = np.expand_dims(np.random.rand(p.shape[0]), axis=1)
            return (p.cumsum(axis=1) > r).argmax(axis=1)

        trajectory = [[] for i in range(num_episodes)]
        done = np.zeros(num_episodes, dtype=np.bool)
        state = np.array([mdp.initial_state] * num_episodes)
        for t in range(max_timesteps):
            action = random_choice_prob_vectorized(p=pi[state, :])
            reward = mdp.R[state, action]
            state1 = random_choice_prob_vectorized(p=mdp.T[state, action, :])
            for i in range(num_episodes):
                if not done[i]:
                    trajectory[i].append((i, t, state[i], action[i], reward[i], state1[i]))
            done = done | (state == mdp.absorbing_state)

            state = state1
    else:
        trajectory = []
        for i in range(num_episodes):
            trajectory_one = []
            state = mdp.initial_state
            for t in range(max_timesteps):
                action = np.random.choice(np.arange(mdp.A), p=pi[state, :])
                reward = mdp.R[state, action]
                state1 = np.random.choice(np.arange(mdp.S), p=mdp.T[state, action, :])

                trajectory_one.append((i, t, state, action, reward, state1))
                if state == mdp.absorbing_state:
                    break
                state = state1
            trajectory.append(trajectory_one)

    return trajectory


def compute_MLE_MDP(S, A, R, gamma, trajectory, absorb_unseen=True):
    N = np.zeros((S, A, S))
    for trajectory_one in trajectory:
        for episode, t, state, action, reward, state1 in trajectory_one:
            N[state, action, state1] += 1

    T = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            if N[s, a, :].sum() == 0:
                if absorb_unseen:
                    T[s, a, S - 1] = 1  # absorbing state
                else:
                    T[s, a, :] = 1. / S
            else:
                T[s, a, :] = N[s, a, :] / N[s, a, :].sum()

    mle_mdp = MDP(S, A, T, R, gamma)

    return mle_mdp, N


def softmax(X, temperature):
    X = np.array(X)
    if len(X.shape) == 2:
        X = (X - np.max(X, axis=1)[:, None]) / (temperature[:, None] + 1e-20)  # TODO
        S = np.exp(X) / (np.sum(np.exp(X), axis=1) + 1e-20)[:, None]
        return S
    elif len(X.shape) == 1:
        X = (X - np.max(X)) / temperature
        S = np.exp(X) / np.sum(np.exp(X))
        return S
    else:
        raise NotImplementedError()


def generate_baseline_policy(seed, mdp, optimality=0.9):
    np.random.seed(seed + 1)
    pi_opt, _, Q_opt = solve_MDP(mdp)
    pi_unif = np.ones((mdp.S, mdp.A)) / mdp.A
    V_opt = policy_evaluation(mdp, pi_opt)[0][0]
    V_unif = policy_evaluation(mdp, pi_unif)[0][0]

    ##################################
    # following SPIBB paper
    ##################################
    V_final_target = V_opt * optimality + (1 - optimality) * V_unif
    V_softmax_target = 0.5 * V_opt + 0.5 * V_final_target
    softmax_reduction_factor = 0.9
    perturbation_reduction_factor = 0.9

    temperature = np.ones(mdp.S) / (2 * 10 ** 6)
    pi_soft = softmax(Q_opt, temperature)
    while policy_evaluation(mdp, pi_soft)[0][0] > V_softmax_target:
        temperature /= softmax_reduction_factor
        pi_soft = softmax(Q_opt, temperature)

    pi_b = pi_soft.copy()
    while policy_evaluation(mdp, pi_b)[0][0] > V_final_target:
        s = np.random.randint(mdp.S)
        a_opt = np.argmax(Q_opt[s, :])
        pi_b[s, a_opt] *= perturbation_reduction_factor
        pi_b[s, :] /= np.sum(pi_b[s, :])
    return pi_b


def compute_marginal_distribution(mdp, pi, regularizer=0):
    """
    d: |S||A|
    """
    p0_s = np.zeros(mdp.S); p0_s[mdp.initial_state] = 1
    p0 = (p0_s[:, None] * pi).reshape(mdp.S * mdp.A)
    P_pi = (mdp.T.reshape(mdp.S * mdp.A, mdp.S)[:, :, None] * pi).reshape(mdp.S * mdp.A, mdp.S * mdp.A)
    d = np.ones(mdp.S * mdp.A); d /= np.sum(d)
    D = np.diag(d)
    E = np.sqrt(D) @ (np.eye(mdp.S * mdp.A) - mdp.gamma * P_pi)

    Q = np.linalg.solve(E.T @ E + regularizer * np.eye(mdp.S * mdp.A), (1 - mdp.gamma) * p0)
    w = Q - mdp.gamma * P_pi @ Q

    assert np.all(w > -1e-3), w
    d_pi = w * d
    d_pi[w < 0] = 0
    d_pi /= np.sum(d_pi)
    return d_pi


def SMODICE(mdp, mdp_expert, pi_star, pi_b, alpha=1., delta=0., example=False):
    d = compute_marginal_distribution(mdp, pi_b)  # |S||A|
    d_s = d.reshape(mdp.S, mdp.A).sum(axis=1) # |S|

    # d_s = d.reshape(mdp.S, mdp.A).sum(axis=1).repeat(mdp.A)
    # different expert distribution depending on whether exampleMDP:
    if not example:
        d_expert = compute_marginal_distribution(mdp_expert, pi_star)
        d_expert_s = d_expert.reshape(mdp_expert.S, mdp_expert.A).sum(axis=1)

        # d_expert_s = d_expert.reshape(mdp_expert.S, mdp_expert.A).sum(axis=1).repeat(mdp_expert.A)
    else:
        # p_U(s|e)
        d_expert_s = np.zeros((mdp.S)) # |S|
        d_expert_s[mdp_expert.absorbing_state] = 1

    p0 = np.zeros(mdp.S); p0[mdp.initial_state] = 1  # |S|
    P = mdp.T.reshape(mdp.S * mdp.A, mdp.S)  # |S||A| x |S|
    R = np.log((d_expert_s+delta)/(d_s+delta)) # |S||A|

    B = np.repeat(np.eye(mdp.S), mdp.A, axis=0)  # |S||A| x |S|
    I = np.ones(mdp.S * mdp.A)
    D = np.diag(d)  # |S||A| x |S||A|
    alpha = alpha

    H = (mdp.gamma * P - B).T @ D @ (mdp.gamma * P - B) # |S| x |S|
    y = -((1 - mdp.gamma) * p0 + (mdp.gamma * P - B).T @ D @ (I + B @ R)) # |S|
    V_star = np.linalg.pinv(H) @ y 

    w_star = B @ R + (mdp.gamma * P - B) @ V_star + 1 
    m = np.array(w_star >= 0, dtype=np.float) # ensure non-negativity
    w_star = w_star * m 

    # Weighted BC
    pi_star = (w_star * d).reshape(mdp.S, mdp.A)

    pi_mle = solve_MDP(mdp)[0]
    for s in range(mdp.S):
        # handle ill-posed state via policy bootstrapping heuristic
        if np.isclose(np.sum(pi_star[s, :]), 0):
            if np.isclose(np.sum(d.reshape(mdp.S, mdp.A)[s, :]), 0):
                pi_star[s, :] = pi_b[s, :]
            else:
                pi_star[s, :] = pi_mle[s, :]
    pi_star /= np.sum(pi_star, axis=1, keepdims=True)

    f_divergence = d.dot(0.5 * (w_star ** 2))

    return pi_star, f_divergence, V_star

