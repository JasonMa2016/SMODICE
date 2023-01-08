import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import wandb

import utils
from smodice_pytorch import SMODICE
from rce_pytorch import RCE_TD3_BC
from oril_pytorch import ORIL

from discriminator_pytorch import Discriminator, Discriminator_SA
np.set_printoptions(precision=3, suppress=True)

def _get_data(env, dataset=None, num_expert_obs=200, terminal_offset=50, all=False, indices=None, skip=1, goal_id=2):
  """Loads the success examples.

  Args:
    env: A PyEnvironment for which we want to generate success examples.
    env_name: The name of the environment.
    num_expert_obs: The number of success examples to generate.
    terminal_offset: For the d4rl datasets, we randomly subsample the last N
      steps to use as success examples. The terminal_offset parameter is N.
  Returns:
    expert_obs: Array with the success examples.
  """

  if dataset is None:
    dataset = env.get_dataset()
  if 'timeouts' in dataset:
    terminals = np.where(dataset['timeouts'])[0][goal_id::skip]
  if 'terminals' in dataset:
    terminals = np.where(dataset['terminals'])[0][goal_id::skip]
  expert_obs = np.concatenate(
      [dataset['observations'][t - terminal_offset:t+1] for t in terminals],
      axis=0)
  expert_infos_qpos = np.concatenate(
      [dataset['infos/qpos'][t - terminal_offset:t+1] for t in terminals],
      axis=0)
  expert_infos_qvel = np.concatenate(
      [dataset['infos/qvel'][t - terminal_offset:t+1] for t in terminals],
      axis=0)
  if not all:
    if indices is None:
      num_expert_obs = min(num_expert_obs, expert_obs.shape[0])
      indices = np.random.choice(
        len(expert_obs), size=num_expert_obs, replace=False)
    expert_obs = expert_obs[indices]
    expert_infos_qpos = expert_infos_qpos[indices]
    expert_infos_qvel = expert_infos_qvel[indices]
  return expert_obs, expert_infos_qpos, expert_infos_qvel

GOAL_MAPPING = {'right':1, 'left':2, 'down':3, 'up':4}
GOAL_LOC_MAPPING = {'right':[7,4], 'left':[1,4], 'down': [4,1], 'up':[4,7]}

def run(config):
    # Load offline dataset
    from envs.pointmaze import maze_model
    maze = maze_model.EXAMPLE_MAZE
    init_target = GOAL_LOC_MAPPING[config['goal']]
    goal_id = GOAL_MAPPING[config['goal']]
    env = maze_model.MazeEnv(maze, reset_target=False, init_target=init_target)
    dataset = utils.get_dataset("envs/demos/maze2d-example-v1-expert.hdf5")

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    env.seed(config['seed'])

    # Load expert dataset
    # skip=4 because the same direction occurs every 4 trajectories
    expert_obs, expert_infos_qpos, expert_infos_qvel = _get_data(env, dataset=dataset, num_expert_obs=200, skip=4, terminal_offset=2, goal_id=goal_id)
    expert_traj = {'observations': expert_obs, 'infos/qpos': expert_infos_qpos, 'infos/qvel': expert_infos_qvel}

    # Process offline dataset
    initial_obs_dataset, dataset, dataset_statistics = utils.dice_dataset(env, standardize_observation=True, absorbing_state=config['absorbing_state'], standardize_reward=config['standardize_reward'], dataset=dataset)

    # Normalize expert observations and potentially add absorbing state
    if config['standardize_obs']:
        expert_obs_dim = expert_traj['observations'].shape[1]
        expert_traj['observations'] = (expert_traj['observations'] - dataset_statistics['observation_mean'][:expert_obs_dim]) / (dataset_statistics['observation_std'][:expert_obs_dim] + 1e-10)
        if 'next_observations' in expert_traj:
            expert_traj['next_observations'] = (expert_traj['next_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
    if config['absorbing_state'] and 'terminal' in expert_traj:
        expert_traj = utils.add_absorbing_state(expert_traj)
    if config['use_policy_entropy_constraint'] or config['use_data_policy_entropy_constraint']:
        if config['target_entropy'] is None:
            config['target_entropy'] = -np.prod(env.action_space.shape)

    # Create inputs for the discriminator
    state_dim = dataset_statistics['observation_dim'] + 1 if config['absorbing_state'] else dataset_statistics['observation_dim']
    action_dim = 0 if config['state'] else dataset_statistics['action_dim']
    disc_cutoff = state_dim

    expert_input = expert_traj['observations'][:, :disc_cutoff]
    offline_input = dataset['observations'][:, :disc_cutoff]

    discriminator = Discriminator_SA(disc_cutoff, action_dim, hidden_dim=config['hidden_sizes'][0], device=config['device'])

    # Train discriminator
    if config['disc_type'] == 'learned':
        dataset_expert = torch.utils.data.TensorDataset(torch.FloatTensor(expert_input))
        expert_loader = torch.utils.data.DataLoader(dataset_expert, batch_size=256, shuffle=True, pin_memory=True)
        dataset_offline = torch.utils.data.TensorDataset(torch.FloatTensor(offline_input))
        offline_loader = torch.utils.data.DataLoader(dataset_offline, batch_size=256, shuffle=True, pin_memory=True)
        for i in range(config['disc_iterations']):
            loss = discriminator.update(expert_loader, offline_loader)
            print(i, loss)

    def _sample_minibatch(batch_size, reward_scale):
        initial_indices = np.random.randint(0, dataset_statistics['N_initial_observations'], batch_size)
        indices = np.random.randint(0, dataset_statistics['N'], batch_size)
        sampled_dataset = (
            initial_obs_dataset['initial_observations'][initial_indices],
            dataset['observations'][indices],
            dataset['actions'][indices],
            dataset['rewards'][indices] * reward_scale,
            dataset['next_observations'][indices],
            dataset['terminals'][indices]
        )
        return tuple(map(torch.from_numpy, sampled_dataset))

    if 'dice' in config['algo_type']:
        agent = SMODICE(dataset_statistics['observation_dim'] + 1 if config['absorbing_state'] else dataset_statistics['observation_dim'],
        dataset_statistics['action_dim'], config=config
        )
    elif 'rce' in config['algo_type']:
        state_dim = dataset_statistics['observation_dim'] + 1 if config['absorbing_state'] else dataset_statistics['observation_dim']
        action_dim = dataset_statistics['action_dim']
        max_action = env.action_space.high[0]
        agent = RCE_TD3_BC(state_dim, action_dim, max_action)
    elif 'oril' in config['algo_type']:
        state_dim = dataset_statistics['observation_dim'] + 1 if config['absorbing_state'] else dataset_statistics['observation_dim']
        action_dim = dataset_statistics['action_dim']
        max_action = env.action_space.high[0]
        agent = ORIL(state_dim, action_dim, max_action)
    else:
        raise NotImplementedError

    result_logs = []
    start_iteration = 0

    # Start training
    start_time = time.time()
    last_start_time = time.time()
    def _eval_and_log(train_result, config):
        nonlocal last_start_time
        train_result = {k: v.detach().cpu().numpy() for k, v in train_result.items()}

        # evaluation via real-env rollout
        eval = utils.evaluate(env, agent, dataset_statistics, absorbing_state=config['absorbing_state'], pid=config.get('pid'),
        iteration=iteration, normalize=False)
        train_result.update({'iteration': iteration, 'eval': eval})

        train_result.update({'iter_per_sec': config['log_iterations'] / (time.time() - last_start_time)})
        if 'w_e' in train_result:
            train_result.update({'w_e': train_result['w_e'].mean()})

        # torch.save(agent._policy_network.state_dict(), f'policy_models/{run_name}-{iteration}')
        result_logs.append({'log': train_result, 'step': iteration})
        if not int(os.environ.get('DISABLE_STDOUT', 0)):
            print(f'=======================================================')
            # for k, v in sorted(train_result.items()):
            #     print(f'- {k:23s}:{v:15.10f}')
            if train_result.get('eval'):
                print(f'- {"eval":23s}:{train_result["eval"]:15.10f}')
            print(f'iteration={iteration} (elapsed_time={time.time() - start_time:.2f}s, {train_result["iter_per_sec"]:.2f}it/s)')
            print(f'=======================================================', flush=True)

        last_start_time = time.time()

    for iteration in tqdm(range(start_iteration, config['total_iterations'] + 1), ncols=70, desc='DICE', initial=start_iteration, total=config['total_iterations'] + 1, ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
        # Sample mini-batch data from dataset
        initial_observation, observation, action, reward, next_observation, terminal = _sample_minibatch(config['batch_size'], config['reward_scale'])

        # Sample success states for RCE
        if config['algo_type'] == 'rce':
            success_indices = np.random.randint(0, expert_traj['observations'].shape[0], config['batch_size'])
            success_state = torch.from_numpy(expert_traj['observations'][success_indices])
            initial_observation = success_state

        # Compute discriminator based reward (SMODICE, ORIL)
        with torch.no_grad():
            obs_for_disc = torch.from_numpy(np.array(observation)).to(discriminator.device)
            if config['state']:
                disc_input = obs_for_disc
            else:
                act_for_disc = torch.from_numpy(np.array(action)).to(discriminator.device)
                disc_input = torch.cat([obs_for_disc, act_for_disc], axis=1)
            reward = discriminator.predict_reward(disc_input)

        # Perform gradient descent
        train_result = agent.train_step(initial_observation, observation, action, reward, next_observation, terminal)
        if iteration % config['log_iterations'] == 0:
            _eval_and_log(train_result, config)

if __name__ == "__main__":
    from configs.oil_examples_pointmass_default import get_parser
    args = get_parser().parse_args()
    run(vars(args))