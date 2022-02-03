import numpy as np
import pickle
import gzip
import h5py
import argparse
from d4rl.locomotion import maze_env, ant, swimmer
from d4rl.locomotion.wrappers import NormalizedBoxEnv
import torch
from PIL import Image
import os


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'timeouts': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, r, tgt, done, timeout, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['timeouts'].append(timeout)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def load_policy(policy_file):
    data = torch.load(policy_file)
    policy = data['exploration/policy'].to('cpu')
    env = data['evaluation/env']
    print("Policy loaded")
    return policy, env

def save_video(save_dir, file_name, frames, episode_id=0):
    filename = os.path.join(save_dir, file_name+ '_episode_{}'.format(episode_id))
    if not os.path.exists(filename):
        os.makedirs(filename)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), 'RGB')
        img.save(os.path.join(filename, 'frame_{}.png'.format(i)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--maze', type=str, default='umaze', help='Maze type. umaze, medium, or large')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    parser.add_argument('--env', type=str, default='ant', help='Environment type')
    parser.add_argument('--policy_file', type=str, default='policy_file', help='file_name')
    parser.add_argument('--max_episode_steps', default=1000, type=int)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--multi_start', action='store_true')
    parser.add_argument('--multigoal', action='store_true')
    args = parser.parse_args()

    if args.maze == 'umaze':
        maze = maze_env.U_MAZE
    elif args.maze == 'medium':
        maze = maze_env.BIG_MAZE
    elif args.maze == 'large':
        maze = maze_env.HARDEST_MAZE
    elif args.maze == 'umaze_eval':
        maze = maze_env.U_MAZE_EVAL
    elif args.maze == 'medium_eval':
        maze = maze_env.BIG_MAZE_EVAL
    elif args.maze == 'large_eval':
        maze = maze_env.HARDEST_MAZE_EVAL
    else:
        raise NotImplementedError
    
    env = NormalizedBoxEnv(ant.AntMazeEnv(maze_map=maze, maze_size_scaling=4.0, non_zero_reset=args.multi_start))
    
    env.set_target()
    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    
    ts = 0
    num_episodes = 0
    for _ in range(args.num_samples):
        act = env.action_space.sample()

        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.2
            act = np.clip(act, -1.0, 1.0)

        ns, r, done, info = env.step(act)
        timeout = False
        if ts >= args.max_episode_steps:
            timeout = True
            #done = True
        
        append_data(data, s[:-2], act, r, env.target_goal, done, timeout, env.physics.data)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1

        if done or timeout:
            done = False
            ts = 0
            s = env.reset()
            env.set_target_goal()
            num_episodes += 1
            frames = []
        else:
            s = ns
    
    fname = 'demos/antmaze-umaze-v2-randomstart-noiserandomaction.hdf5'
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == '__main__':
    main()
