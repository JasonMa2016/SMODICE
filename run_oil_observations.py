import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch
from tqdm import tqdm

from smodice_pytorch import SMODICE
from oril_pytorch import ORIL 

from discriminator_pytorch import Discriminator, Discriminator_SA
import utils

np.set_printoptions(precision=3, suppress=True)

MUJOCO = ['hopper', 'walker2d', 'halfcheetah', 'ant']

def run(config):
    version = 'v2'
    if 'kitchen' in config['env_name']:
        version  = 'v0'

    # Load environment
    if not config['mismatch']:
        env = gym.make(f"{config['env_name']}-{config['dataset']}-{version}")
    else:      
        env = gym.make(f"{config['env_name']}-random-{version}")

    if config['env_name'] not in MUJOCO:
        if config['env_name'] != 'kitchen':
            expert_env = gym.make(f"{config['env_name']}-{config['dataset']}-{version}")
        else:
            expert_env = gym.make(f"kitchen-complete-v0")
    else:
        expert_env = gym.make(f"{config['env_name']}-expert-{version}")
    
    # Seeding
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 
    env.seed(config['seed'])
    expert_env.seed(config['seed'])

    # Load expert dataset
    if not config['mismatch']:
        traj_iterator = utils.sequence_dataset(expert_env)
        expert_traj = next(traj_iterator)
    else:
        # Load mismatch expert dataset 
        demo_file = f"envs/demos/{config['env_name']}_{config['dataset']}.pkl"
        demo = pickle.load(open(demo_file, 'rb'))
        if 'ant' in config['env_name']: 
            expert_obs = np.array(demo['observations'][:1000])
            expert_actions = np.array(demo['actions'][:1000])
            expert_next_obs = np.array(demo['next_observations'][:1000])
        else:
            expert_obs = np.array(demo['observations'][0])
            expert_actions = np.array(demo['actions'][0])
            expert_next_obs = np.array(demo['next_observations'][0])
        expert_traj = {'observations': expert_obs, 'actions': expert_actions, 'next_observations': expert_next_obs}

    # Load offline dataset
    if config['num_expert_traj'] == 0:
        initial_obs_dataset, dataset, dataset_statistics = utils.dice_dataset(env, standardize_observation=config['standardize_obs'], absorbing_state=config['absorbing_state'], standardize_reward=config['standardize_reward'])
    else:
        initial_obs_dataset, dataset, dataset_statistics = utils.dice_combined_dataset(expert_env, env, num_expert_traj=config['num_expert_traj'], num_offline_traj=config['num_offline_traj'],
    standardize_observation=config['standardize_obs'], absorbing_state=config['absorbing_state'],
     standardize_reward=config['standardize_reward'])

    # Normalize expert observations and potentially add absorbing state
    if config['standardize_obs']:
        expert_obs_dim = expert_traj['observations'].shape[1]
        expert_traj['observations'] = (expert_traj['observations'] - dataset_statistics['observation_mean'][:expert_obs_dim]) / (dataset_statistics['observation_std'][:expert_obs_dim] + 1e-10)
        if 'next_observations' in expert_traj:
            expert_traj['next_observations'] = (expert_traj['next_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
    if config['absorbing_state']:
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
        for i in tqdm(range(config['disc_iterations'])):
            loss = discriminator.update(expert_loader, offline_loader)

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

    # Intialize SMODICE policy
    if 'smodice' in config['algo_type']:
        agent = SMODICE(
            observation_spec=dataset_statistics['observation_dim'] + 1 if config['absorbing_state'] else dataset_statistics['observation_dim'],
            action_spec=dataset_statistics['action_dim'],
            config=config
        )
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
    for iteration in tqdm(range(start_iteration, config['total_iterations'] + 1), ncols=70, desc='SMODICE', initial=start_iteration, total=config['total_iterations'] + 1, ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
        # Sample mini-batch data from dataset
        initial_observation, observation, action, reward, next_observation, terminal, expert = _sample_minibatch(config['batch_size'], config['reward_scale'])
        
        # Get rewards
        with torch.no_grad():
            obs_for_disc = torch.from_numpy(np.array(observation)).to(discriminator.device)
            if config['state']:
                disc_input = obs_for_disc
            else:
                act_for_disc = torch.from_numpy(np.array(action)).to(discriminator.device)
                disc_input = torch.cat([obs_for_disc, act_for_disc], axis=1)
            reward = discriminator.predict_reward(disc_input)

            # Zero-reward ablation
            if config['disc_type'] == 'zero':
                reward = torch.zeros_like(reward)

        # Perform gradient descent
        max_steps = 280 if 'kitchen' in config['env_name'] else None 
        train_result = agent.train_step(initial_observation, observation, action, reward, next_observation, terminal)
        
        # Logging
        if iteration % config['log_iterations'] == 0:
            train_result = {k: v.cpu().detach().numpy() for k, v in train_result.items()}
            # evaluation via real-env rollout
            eval = utils.evaluate(env, agent, dataset_statistics, absorbing_state=config['absorbing_state'], 
            iteration=iteration, max_steps=max_steps)
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

                if train_result.get('eval'):
                    print(f'- {"eval":23s}:{train_result["eval"]:15.10f}')
                print(f'iteration={iteration} (elapsed_time={time.time() - start_time:.2f}s, {train_result["iter_per_sec"]:.2f}it/s)')
                print(f'=======================================================', flush=True)

            last_start_time = time.time()


if __name__ == "__main__":
    from configs.oil_observations_default import get_parser
    args = get_parser().parse_args()

    if args.env_name == 'walker2d':
        args.num_expert_traj = 100

    # This is just the kitchen environment
    if args.env_name == 'kitchen':
        args.num_expert_traj = 0
        args.num_offline_traj = 2000
        args.absorbing_state = False 
        args.f = 'chi'
        args.dataset = 'mixed'

    if args.mismatch == True:
        args.absorbing_state = False         
    
    run(vars(args))

