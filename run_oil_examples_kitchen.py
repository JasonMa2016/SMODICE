import os
import time

import d4rl
import gym
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image 

import torch
from tqdm import tqdm
import wandb 

import envs 
import utils
from smodice_pytorch import SMODICE
from rce_pytorch import RCE_TD3_BC
from oril_pytorch import ORIL 
from discriminator_pytorch import Discriminator_SA

np.set_printoptions(precision=3, suppress=True)


def run(config):
    # load offline dataset    
    env = gym.make(f"{config['env_name']}-mixed-v0")
    dataset = env.get_dataset()

    # Load the custom kitchen environment that gets the success examples
    evaluation_env = gym.make(f"{config['env_name']}-{config['dataset']}-v0")
    expert_obs = evaluation_env.get_example(dataset, num_expert_obs=500)
    expert_traj = {'observations': expert_obs}
    
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    env.seed(config['seed'])
    evaluation_env.seed(config['seed'])

    initial_obs_dataset, dataset, dataset_statistics = utils.dice_dataset(env, standardize_observation=config['standardize_obs'], absorbing_state=config['absorbing_state'], standardize_reward=config['standardize_reward'])

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

        # Train discriminator
        print("Train Discriminator")
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
            dataset['terminals'][indices]
        )
        return tuple(map(torch.from_numpy, sampled_dataset))

    def _evaluate(env, agent, dataset_statistics, absorbing_state=True, num_evaluation=10, normalize=False, make_gif=False, iteration=0, max_steps=None, run_name=''):
        normalized_scores = []
        if max_steps is None:
            max_steps = env._max_episode_steps
        imgs = []
        for eval_iter in range(num_evaluation):
            start_time = time.time()
            obs = env.reset()
            episode_reward = 0
            for t in tqdm(range(max_steps), ncols=70, desc='evaluate', ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
                if absorbing_state:
                    obs_standardized = np.append((obs - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10), 0)
                else:
                    obs_standardized = (obs - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)

                actions = agent.step((np.array([obs_standardized])).astype(np.float32))
                action = actions[0][0].numpy()
                
                # prevent NAN
                action = np.clip(action, env.action_space.low, env.action_space.high)
                next_obs, reward, done, info = env.step(action)
                
                # only care about the specified task
                if config['dataset'] in info['rewards']['removed']:
                    reward = 1.0
                    done = True
                else:
                    reward = 0.0 
                    done = False  

                if make_gif and eval_iter == 0:
                    img = env.render(mode="rgb_array")
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
        
        # Evaluation
        if iteration % config['log_iterations'] == 0:
            train_result = {k: v.detach().cpu().numpy() for k, v in train_result.items()}
            # evaluation via real-env rollout
            eval = _evaluate(env, agent, dataset_statistics, absorbing_state=config['absorbing_state'],
                normalize=False, num_evaluation=10, max_steps=280)
            train_result.update({'iteration': iteration, 'eval': eval})

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
                print(f'iteration={iteration} (elapsed_time={time.time() - start_time:.2f}s, {train_result["iter_per_sec"]:.2f}it/s)')
                print(f'=======================================================', flush=True)

            last_start_time = time.time()
            

if __name__ == "__main__":
    from configs.oil_examples_kitchen_default import get_parser
    args = get_parser().parse_args()
    run(vars(args))