from .kitchen_envs import KitchenMicrowaveV0, KitchenKettleV0
from gym.envs.registration import register

register(
    id='kitchen-microwave-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)

register(
    id='kitchen-kettle-v0',
    entry_point='d4rl.kitchen:KitchenKettleV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)




