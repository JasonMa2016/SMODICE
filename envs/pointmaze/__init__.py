from .maze_model import MazeEnv, OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE, U_MAZE_EVAL, MEDIUM_MAZE_EVAL, LARGE_MAZE_EVAL
from gym.envs.registration import register

register(
    id='point-open-v0',
    entry_point='pointmaze:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'maze_spec':OPEN,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 0.01,
        'ref_max_score': 20.66,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-sparse.hdf5'
    }
)

register(
    id='point-umaze-v0',
    entry_point='pointmaze:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 0.94,
        'ref_max_score': 62.6,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse.hdf5'
    }
)

register(
    id='point-medium-v0',
    entry_point='pointmaze:MazeEnv',
    max_episode_steps=250,
    kwargs={
        'maze_spec':MEDIUM_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 5.77,
        'ref_max_score': 85.14,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse.hdf5'
    }
)


register(
    id='point-large-v0',
    entry_point='pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':LARGE_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse.hdf5'
    }
)


register(
    id='point-umaze-v1',
    entry_point='pointmaze:MazeEnv',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)

register(
    id='point-medium-v1',
    entry_point='pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MEDIUM_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5'
    }
)


register(
    id='point-large-v1',
    entry_point='pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec':LARGE_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 6.7,
        'ref_max_score': 273.99,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
    }
)

register(
    id='point-eval-umaze-v1',
    entry_point='pointmaze:MazeEnv',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE_EVAL,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 36.63,
        'ref_max_score': 141.4,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-umaze-sparse-v1.hdf5'
    }
)

register(
    id='point-eval-medium-v1',
    entry_point='pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MEDIUM_MAZE_EVAL,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.07,
        'ref_max_score': 204.93,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-medium-sparse-v1.hdf5'
    }
)


register(
    id='point-eval-large-v1',
    entry_point='pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec':LARGE_MAZE_EVAL,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 16.4,
        'ref_max_score': 302.22,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-large-sparse-v1.hdf5'
    }
)

