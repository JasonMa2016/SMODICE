import os
import time

import glob 
import d4rl
import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image 
import torch
from tqdm import tqdm
import wandb
from smodice_pytorch import SMODICE
from rce_pytorch import RCE_TD3_BC
from oril_pytorch import ORIL 

import utils
from discriminator_pytorch import Discriminator, Discriminator_SA

np.set_printoptions(precision=3, suppress=True)

def _get_data(env, dataset=None, num_expert_obs=200, terminal_offset=50, all=False, indices=None, skip=1):
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
    terminals = np.where(dataset['timeouts'])[0][::skip]
  if 'terminals' in dataset:
    terminals = np.where(dataset['terminals'])[0][::skip]
  expert_obs = np.concatenate(
      [dataset['observations'][t - terminal_offset:t] for t in terminals],
      axis=0)
  if not all:
    if indices is None:
      num_expert_obs = min(num_expert_obs, expert_obs.shape[0])
      indices = np.random.choice(
        len(expert_obs), size=num_expert_obs, replace=False)
    expert_obs = expert_obs[indices]
  return expert_obs


def run(config):
    env = gym.make(f"{config['env_name']}-{config['dataset']}-v2")
    env.reset()
    dataset = utils.get_dataset("envs/demos/antmaze-umaze-v2-randomstart-noiserandomaction.hdf5")
    expert_env = gym.make(f"antmaze-umaze-v2")

    if config['mismatch']:
        # Use pointmass as demonstrator
        expert_dataset = utils.get_dataset("envs/demos/maze2d-umaze-v1-expert.hdf5")
    else:
        expert_dataset = expert_env.get_dataset()  
        # Uses examples of success outcomes 
        if config['example']:   
            expert_obs = _get_data(expert_env, num_expert_obs=500, terminal_offset=10)
            expert_dataset['observations'] = expert_obs
    
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    env.seed(config['seed'])
    expert_env.seed(config['seed'])

    # the "expert_traj" here is just the original antmaze dataset (e.g., antmaze-umaze-v2), which contains some expert-level data
    initial_obs_dataset, dataset, dataset_statistics = utils.dice_combined_dataset(expert_env, env, offline_dataset=dataset,
    num_expert_traj=config['num_expert_traj'], num_offline_traj=config['num_offline_traj'],
    standardize_observation=config['standardize_obs'], absorbing_state=config['absorbing_state'], standardize_reward=config['standardize_reward'],
    reverse=config['reverse'])
    
    # Normalize expert observations and potentially add absorbing state
    if config['standardize_obs']:
        expert_obs_dim = expert_dataset['observations'].shape[1]
        expert_dataset['observations'] = (expert_dataset['observations'] - dataset_statistics['observation_mean'][:expert_obs_dim]) / (dataset_statistics['observation_std'][:expert_obs_dim] + 1e-10)
        if 'next_observations' in expert_dataset:
            expert_dataset['next_observations'] = (expert_dataset['next_observations'] - dataset_statistics['observation_mean'][:expert_obs_dim]) / (dataset_statistics['observation_std'][:expert_obs_dim] + 1e-10)
    
    # Normalize to xy-coordinates to [0,1]
    expert_min = expert_dataset['observations'][:, :2].min(axis=0)
    expert_max = expert_dataset['observations'][:, :2].max(axis=0)
    offline_min = dataset['observations'][:, :2].min(axis=0)
    offline_max = dataset['observations'][:, :2].max(axis=0)

    if config['mismatch']:
        expert_dataset['observations'][:, :2] = (expert_dataset['observations'][:, :2] - expert_min) / (expert_max - expert_min)
    else:
        expert_dataset['observations'][:, :2] = (expert_dataset['observations'][:, :2] - offline_min) / (offline_max - offline_min)

    dataset['observations'][:, :2] = (dataset['observations'][:, :2] - offline_min) / (offline_max - offline_min)
    if 'next_observations' in dataset:
        dataset['next_observations'][:, :2] = (dataset['next_observations'][:, :2] - offline_min) / (offline_max - offline_min)
    if 'next_observations' in expert_dataset:
        if config['mismatch']:
            expert_dataset['next_observations'][:, :2] = (expert_dataset['next_observations'][:, :2] - expert_min) / (expert_max - expert_min)
        else:
            expert_dataset['next_observations'][:, :2] = (expert_dataset['next_observations'][:, :2] - offline_min) / (offline_max - offline_min)

    initial_obs_dataset['initial_observations'][:, :2] = (initial_obs_dataset['initial_observations'][:, :2] - offline_min) / (offline_max - offline_min)

    # do some rotation and reflection to line up ant and pointmass data
    if config['mismatch']:
        angle = -np.pi/2
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        reflection = np.array([[1., 0.], [0., -1.]])
        expert_dataset['observations'][:, :2] = np.dot(np.dot(expert_dataset['observations'][:, :2], rotation.T), reflection.T)
    if config['absorbing_state']:
        expert_dataset = utils.add_absorbing_state(expert_dataset)

    traj_iterator = utils.sequence_dataset(expert_env, dataset=expert_dataset, include_next_obs=False)
    
    # Find the correct expert trajectory
    if not config['example']:
        expert_id = 9 if not config['mismatch'] else 1
        for i in range(expert_id):
            expert_traj = next(traj_iterator)
    else:
        expert_traj = {'observations': expert_dataset['observations']}

    if config['use_policy_entropy_constraint'] or config['use_data_policy_entropy_constraint']:
        if config['target_entropy'] is None:
            config['target_entropy'] = -np.prod(env.action_space.shape)
    
    # Create inputs for the discriminator
    state_dim = dataset_statistics['observation_dim'] + 1 if config['absorbing_state'] else dataset_statistics['observation_dim']
    action_dim = 0 if config['state'] else dataset_statistics['action_dim']
    disc_cutoff = 2 if config['mismatch'] else state_dim 
        
    if config['state']:
        expert_input = expert_traj['observations'][:, :disc_cutoff]
        offline_input = dataset['observations'][:, :disc_cutoff]
    else:
        expert_input = np.concatenate((expert_traj['observations'][:, :disc_cutoff], expert_traj['actions']),axis=1)
        offline_input = np.concatenate((dataset['observations'][:, :disc_cutoff], dataset['actions']),axis=1)

    discriminator = Discriminator_SA(disc_cutoff, action_dim, hidden_dim=config['hidden_sizes'][0], device=config['device'])

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
            dataset['terminals'][indices],
            dataset['experts'][indices]
        )
        return tuple(map(torch.from_numpy, sampled_dataset))

    def _evaluate(env, agent, dataset_statistics, absorbing_state=False,
     num_evaluation=10, pid=None, normalize=True, make_gif=False, iteration=0):
        normalized_scores = []

        imgs = []
        for eval_iter in range(num_evaluation):
            start_time = time.time()
            obs = env.reset()
            episode_reward = 0
            for t in tqdm(range(env._max_episode_steps), ncols=70, desc='evaluate', ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
                if absorbing_state:
                    obs_standardized = np.append((obs - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10), 0)
                else:
                    obs_standardized = (obs - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
                obs_standardized[:2] = (obs_standardized[:2] - offline_min) / (offline_max - offline_min)
                actions = agent.step((np.array([obs_standardized])).astype(np.float32))
                action = actions[0][0].numpy()
                
                # prevent NAN
                action = np.clip(action, env.action_space.low, env.action_space.high)
                next_obs, reward, done, info = env.step(action)
                if make_gif and eval_iter == 0:
                    img = env.render(mode='rgb_array')
                    imgs.append(Image.fromarray(img))
                episode_reward += reward
                if done:
                    break
                obs = next_obs
            if normalize:
                normalized_score = 100 * (episode_reward - d4rl.infos.REF_MIN_SCORE[env.spec.id]) / (d4rl.infos.REF_MAX_SCORE[env.spec.id] - d4rl.infos.REF_MIN_SCORE[env.spec.id])
            else:
                normalized_score = episode_reward
            print(f'normalized_score: {normalized_score} (elapsed_time={time.time() - start_time:.3f}) ')
            normalized_scores.append(normalized_score)

        if make_gif:
            imgs = np.array(imgs)
            imgs[0].save(f"policy_gifs/{run_name}-iter{iteration}.gif", save_all=True,
                append_images=imgs[1:], duration=60, loop=0)
        return np.mean(normalized_scores)

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
    for iteration in tqdm(range(start_iteration, config['total_iterations'] + 1), ncols=70, desc='DICE', initial=start_iteration, total=config['total_iterations'] + 1, ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
        # Sample mini-batch data from dataset
        initial_observation, observation, action, reward, next_observation, terminal, expert = _sample_minibatch(config['batch_size'], config['reward_scale'])
        
        # Sample success states for RCE
        if config['algo_type'] == 'rce':
            success_indices = np.random.randint(0, expert_traj['observations'].shape[0], config['batch_size'])
            success_state = torch.from_numpy(expert_traj['observations'][success_indices])
            initial_observation = success_state
        
        # Compute discriminator based reward (SMODICE, ORIL)
        with torch.no_grad():
            obs_for_disc = torch.from_numpy(np.array(observation[:, :disc_cutoff])).to(discriminator.device)
            if config['state']:
                disc_input = obs_for_disc
            else:
                act_for_disc = torch.from_numpy(np.array(action)).to(discriminator.device)
                disc_input = torch.cat([obs_for_disc, act_for_disc], axis=1)
            reward = discriminator.predict_reward(disc_input)
            if config['disc_type'] == 'zero':
                reward = torch.zeros_like(reward)

        # Perform gradient descent
        train_result = agent.train_step(initial_observation, observation, action, reward, next_observation, terminal)
        if iteration % config['log_iterations'] == 0:
            train_result = {k: v.cpu().detach().numpy() for k, v in train_result.items()}
            # evaluation via real-env rollout
            eval = _evaluate(env, agent, dataset_statistics, absorbing_state=config['absorbing_state'])
            train_result.update({'iteration': iteration, 'eval': eval})

            # compute the important-weights for expert vs. offline data
            expert_index = (expert==1).nonzero(as_tuple=False)
            offline_index = (expert==0).nonzero(as_tuple=False)
            if 'w_e' in train_result:
                w_e = train_result['w_e']
                w_e_expert = w_e[expert_index].mean()
                w_e_offline = w_e[offline_index].mean()
                w_e_ratio = w_e_expert / w_e_offline
                w_e_overall = w_e.mean()
                train_result.update({'w_e': w_e_overall, 'w_e_expert': w_e_expert, 'w_e_offline': w_e_offline, 'w_e_ratio': w_e_ratio})

            train_result.update({'iter_per_sec': config['log_iterations'] / (time.time() - last_start_time)})
            if 'w_e' in train_result:
                train_result.update({'w_e': train_result['w_e'].mean()})
            result_logs.append({'log': train_result, 'step': iteration})
            if not int(os.environ.get('DISABLE_STDOUT', 0)):
                print(f'=======================================================')
                # for k, v in sorted(train_result.items()):
                #     print(f'- {k:23s}:{v:15.10f}')
                if train_result.get('eval'):
                    print(f'- {"eval":23s}:{train_result["eval"]:15.10f}')
                # print(f'config={config}')
                print(f'iteration={iteration} (elapsed_time={time.time() - start_time:.2f}s, {train_result["iter_per_sec"]:.2f}it/s)')
                print(f'=======================================================', flush=True)

            last_start_time = time.time()


if __name__ == "__main__":
    from configs.oil_antmaze_default import get_parser
    args = get_parser().parse_args()

    run(vars(args))

