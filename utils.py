import collections
import h5py 
import os
from PIL import Image
import time
import wandb 

import d4rl
import numpy as np
from tqdm import tqdm

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def dice_dataset(env, standardize_observation=True, absorbing_state=True, standardize_reward=True, dataset=None):
    """
    env: d4rl environment
    """
    if dataset is None:
        dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]
    initial_obs_, obs_, next_obs_, action_, reward_, done_, expert_ = [], [], [], [], [], [], []

    use_timeouts = ('timeouts' in dataset)

    episode_step = 0
    reverse_current_traj = False
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        is_final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == env._max_episode_steps - 1)
        if is_final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue

        if episode_step == 0:
            initial_obs_.append(obs)

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        expert_.append(bool(0)) # assume not expert
        episode_step += 1

        if done_bool or is_final_timestep:
            episode_step = 0

    initial_obs_dataset = {
        'initial_observations': np.array(initial_obs_, dtype=np.float32)
    }
    dataset = {
        'observations': np.array(obs_, dtype=np.float32),
        'actions': np.array(action_, dtype=np.float32),
        'next_observations': np.array(next_obs_, dtype=np.float32),
        'rewards': np.array(reward_, dtype=np.float32),
        'terminals': np.array(done_, dtype=np.float32),
        'experts': np.array(expert_, dtype=np.float32)
    }
    dataset_statistics = {
        'observation_mean': np.mean(dataset['observations'], axis=0),
        'observation_std': np.std(dataset['observations'], axis=0),
        'reward_mean': np.mean(dataset['rewards']),
        'reward_std': np.std(dataset['rewards']),
        'N_initial_observations': len(initial_obs_),
        'N': len(obs_),
        'observation_dim': dataset['observations'].shape[-1],
        'action_dim': dataset['actions'].shape[-1]
    }

    if standardize_observation:
        initial_obs_dataset['initial_observations'] = (initial_obs_dataset['initial_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
        dataset['observations'] = (dataset['observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
        dataset['next_observations'] = (dataset['next_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
    if standardize_reward:
        dataset['rewards'] = (dataset['rewards'] - dataset_statistics['reward_mean']) / (dataset_statistics['reward_std'] + 1e-10)

    if absorbing_state:
        # add additional dimension to observations to deal with absorbing state
        initial_obs_dataset['initial_observations'] = np.concatenate((initial_obs_dataset['initial_observations'], np.zeros((dataset_statistics['N_initial_observations'], 1))), axis=1).astype(np.float32)
        dataset['observations'] = np.concatenate((dataset['observations'], np.zeros((dataset_statistics['N'], 1))), axis=1).astype(np.float32)
        dataset['next_observations'] = np.concatenate((dataset['next_observations'], np.zeros((dataset_statistics['N'], 1))), axis=1).astype(np.float32)
        terminal_indices = np.where(dataset['terminals'])[0]
        absorbing_state = np.eye(dataset_statistics['observation_dim'] + 1)[-1].astype(np.float32)
        dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
            list(dataset['observations']), list(dataset['actions']), list(dataset['rewards']), list(dataset['next_observations']), list(dataset['terminals'])
        for terminal_idx in terminal_indices:
            dataset['next_observations'][terminal_idx] = absorbing_state
            dataset['observations'].append(absorbing_state)
            dataset['actions'].append(dataset['actions'][terminal_idx])
            dataset['rewards'].append(0)
            dataset['next_observations'].append(absorbing_state)
            dataset['terminals'].append(1)

        dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
            np.array(dataset['observations'], dtype=np.float32), np.array(dataset['actions'], dtype=np.float32), np.array(dataset['rewards'], dtype=np.float32), \
            np.array(dataset['next_observations'], dtype=np.float32), np.array(dataset['terminals'], dtype=np.float32)

    return initial_obs_dataset, dataset, dataset_statistics


def dice_combined_dataset(expert_env, env, num_expert_traj=2000, num_offline_traj=2000, expert_dataset=None, offline_dataset=None,
                            standardize_observation=True, absorbing_state=True, standardize_reward=True, reverse=False):
    """
    env: d4rl environment
    """
    initial_obs_, obs_, next_obs_, action_, reward_, done_, expert_ = [], [], [], [], [], [], []

    def add_data(env, num_traj, dataset=None, expert_data=False):
        if dataset is None:
            dataset = env.get_dataset()
        N = dataset['rewards'].shape[0]
        use_timeouts = ('timeouts' in dataset)
        traj_count = 0
        episode_step = 0
        reverse_current_traj = 0
        for i in range(N-1):
            # only use this condition when num_traj < 2000
            if num_traj != 2000 and traj_count == num_traj:
                break
            obs = dataset['observations'][i].astype(np.float32)
            new_obs = dataset['observations'][i+1].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            done_bool = bool(dataset['terminals'][i])

            is_final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == env._max_episode_steps - 1)
            if is_final_timestep:
                # Skip this transition and don't apply terminals on the last step of an episode
                traj_count += 1
                episode_step = 0
                reverse_current_traj = not reverse_current_traj
                continue

            if episode_step == 0:
                initial_obs_.append(obs)

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            expert_.append(expert_data)
            episode_step += 1

            if done_bool or is_final_timestep:
                traj_count += 1
                episode_step = 0
                reverse_current_traj = not reverse_current_traj

    add_data(expert_env, num_expert_traj, dataset=expert_dataset, expert_data=True)
    expert_size = len(obs_)
    print(f"Expert Traj {num_expert_traj}, Expert Size {expert_size}")
    add_data(env, num_offline_traj, dataset=offline_dataset, expert_data=False)
    offline_size = len(obs_) - expert_size 
    print(f"Offline Traj {num_offline_traj}, Offline Size {offline_size}")
    
    initial_obs_dataset = {
        'initial_observations': np.array(initial_obs_, dtype=np.float32)
    }
    dataset = {
        'observations': np.array(obs_, dtype=np.float32),
        'actions': np.array(action_, dtype=np.float32),
        'next_observations': np.array(next_obs_, dtype=np.float32),
        'rewards': np.array(reward_, dtype=np.float32),
        'terminals': np.array(done_, dtype=np.float32),
        'experts': np.array(expert_, dtype=np.float32)
    }
    dataset_statistics = {
        'observation_mean': np.mean(dataset['observations'], axis=0),
        'observation_std': np.std(dataset['observations'], axis=0),
        'reward_mean': np.mean(dataset['rewards']),
        'reward_std': np.std(dataset['rewards']),
        'N_initial_observations': len(initial_obs_),
        'N': len(obs_),
        'observation_dim': dataset['observations'].shape[-1],
        'action_dim': dataset['actions'].shape[-1]
    }

    if standardize_observation:
        initial_obs_dataset['initial_observations'] = (initial_obs_dataset['initial_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
        dataset['observations'] = (dataset['observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
        dataset['next_observations'] = (dataset['next_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
    if standardize_reward:
        dataset['rewards'] = (dataset['rewards'] - dataset_statistics['reward_mean']) / (dataset_statistics['reward_std'] + 1e-10)

    if absorbing_state:
        # add additional dimension to observations to deal with absorbing state
        initial_obs_dataset['initial_observations'] = np.concatenate((initial_obs_dataset['initial_observations'], np.zeros((dataset_statistics['N_initial_observations'], 1))), axis=1).astype(np.float32)
        dataset['observations'] = np.concatenate((dataset['observations'], np.zeros((dataset_statistics['N'], 1))), axis=1).astype(np.float32)
        dataset['next_observations'] = np.concatenate((dataset['next_observations'], np.zeros((dataset_statistics['N'], 1))), axis=1).astype(np.float32)
        terminal_indices = np.where(dataset['terminals'])[0]
        absorbing_state = np.eye(dataset_statistics['observation_dim'] + 1)[-1].astype(np.float32)
        dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
            list(dataset['observations']), list(dataset['actions']), list(dataset['rewards']), list(dataset['next_observations']), list(dataset['terminals'])
        for terminal_idx in terminal_indices:
            dataset['next_observations'][terminal_idx] = absorbing_state
            dataset['observations'].append(absorbing_state)
            dataset['actions'].append(dataset['actions'][terminal_idx])
            dataset['rewards'].append(0)
            dataset['next_observations'].append(absorbing_state)
            dataset['terminals'].append(1)

        dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
            np.array(dataset['observations'], dtype=np.float32), np.array(dataset['actions'], dtype=np.float32), np.array(dataset['rewards'], dtype=np.float32), \
            np.array(dataset['next_observations'], dtype=np.float32), np.array(dataset['terminals'], dtype=np.float32)

    return initial_obs_dataset, dataset, dataset_statistics


def evaluate(env, agent, dataset_statistics, absorbing_state=True, num_evaluation=10, pid=None, normalize=True, make_gif=False, iteration=0, max_steps=None, run_name=''):
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
        if pid is not None:
            print(f'PID [{pid}], Eval Iteration {eval_iter}')
        print(f'normalized_score: {normalized_score} (elapsed_time={time.time() - start_time:.3f}) ')
        normalized_scores.append(normalized_score)

    if make_gif:
        imgs = np.array(imgs)
        imgs[0].save(f"policy_gifs/{run_name}-iter{iteration}.gif", save_all=True,
            append_images=imgs[1:], duration=30, loop=0)
    # print(normalized_scores)
    return np.mean(normalized_scores)


def sequence_dataset(env, dataset=None, sparse=False, **kwargs):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        sparse: if set True, return a trajectory where sparse reward of 1 is attained.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    fields = ['actions', 'observations', 'rewards', 'terminals']
    if 'infos/qpos' in dataset:
        fields.append('infos/qpos')
        fields.append('infos/qvel')
    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True
        fields.append('timeouts')

    episode_step = 0
    if 'next_observations' in dataset.keys():
        fields.append('next_observations')

    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in fields:
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            if sparse:
                if 1 in episode_data['rewards']:
                    yield episode_data
                else:
                    continue
            else:
                yield episode_data 
            data_ = collections.defaultdict(list)

        episode_step += 1

def add_absorbing_state(dataset):
    N = dataset['observations'].shape[0]
    obs_dim = dataset['observations'].shape[1]
    dataset['observations'] = np.concatenate((dataset['observations'], np.zeros((N, 1))), axis=1).astype(np.float32)
    dataset['next_observations'] = np.concatenate((dataset['next_observations'], np.zeros((N, 1))), axis=1).astype(np.float32)
    terminal_indices = np.where(dataset['terminals'])[0]
    absorbing_state = np.eye(obs_dim + 1)[-1].astype(np.float32)
    dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
        list(dataset['observations']), list(dataset['actions']), list(dataset['rewards']), list(dataset['next_observations']), list(dataset['terminals'])
    for terminal_idx in terminal_indices:
        dataset['next_observations'][terminal_idx] = absorbing_state
        dataset['observations'].append(absorbing_state)
        dataset['actions'].append(dataset['actions'][terminal_idx])
        dataset['rewards'].append(0)
        dataset['next_observations'].append(absorbing_state)
        dataset['terminals'].append(1)

    dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
        np.array(dataset['observations'], dtype=np.float32), np.array(dataset['actions'], dtype=np.float32), np.array(dataset['rewards'], dtype=np.float32), \
        np.array(dataset['next_observations'], dtype=np.float32), np.array(dataset['terminals'], dtype=np.float32) 
    return dataset   

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def get_dataset(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    return data_dict

def makedir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path