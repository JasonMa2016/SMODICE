import argparse
from configs.config_utils import boolean

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='pointmass-4direction-v0', type=str)
    parser.add_argument('--goal', default='left', type=str)
    parser.add_argument('--state', default=True, type=boolean)
    parser.add_argument('--algo_type', default='smodice', type=str)
    parser.add_argument('--disc_type', default='learned', type=str)  
    parser.add_argument('--gamma', default=0.99, type=float)

    parser.add_argument('--total_iterations', default=int(1e6), type=int)
    parser.add_argument('--disc_iterations', default=int(1e3), type=int)
    parser.add_argument('--log_iterations', default=int(5e3), type=int)

    parser.add_argument('--absorbing_state', default=False, type=boolean)
    parser.add_argument('--standardize_reward', default=True, type=boolean)
    parser.add_argument('--standardize_obs', default=True, type=boolean)
    parser.add_argument('--reward_scale', default=0.1, type=float)
    parser.add_argument('--mean_range', default=(-7.24, 7.24))
    parser.add_argument('--logstd_range', default=(-5., 2.))

    parser.add_argument('--hidden_sizes', default=(256, 256))
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--resume', default=False, type=boolean)
    parser.add_argument('--f', default='chi', type=str)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--actor_lr', default=3e-5, type=float)
    parser.add_argument('--v_l2_reg', default=0.0001, type=float)
    parser.add_argument('--use_policy_entropy_constraint', default=True, type=boolean)
    parser.add_argument('--target_entropy', default=None, type=float)
    
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--wandb', default=False, type=boolean)
    parser.add_argument('--make_gif', default=False, type=boolean)
    parser.add_argument('--seed', default=0, type=int)

    return parser