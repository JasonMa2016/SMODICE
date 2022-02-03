"""Environments using kitchen and Franka robot."""
import os
import numpy as np
from d4rl.kitchen.adept_envs.utils.configurable import configurable
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

from d4rl.offline_env import OfflineEnv
from dm_control.mujoco import engine

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }
BONUS_THRESH = 0.3

@configurable(pickleable=True)
class KitchenBase(KitchenTaskRelaxV1, OfflineEnv):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True

    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        super(KitchenBase, self).__init__(**kwargs)
        OfflineEnv.__init__(
            self,
            dataset_url=dataset_url,
            ref_max_score=ref_max_score,
            ref_min_score=ref_min_score)

    def _get_task_goal(self):
        new_goal = np.zeros_like(self.goal)
        for element in self.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        next_goal = obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
            complete = distance < BONUS_THRESH
            if complete:
                completions.append(element)
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            # print("remove ever")
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict['bonus'] = bonus
        reward_dict['r_total'] = bonus
        reward_dict['removed'] = completions
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        return obs, reward, done, env_info
    
    def get_example(self, dataset, num_expert_obs=500):
        obs = dataset['observations']
        # this function should only be called for offline-IL from example setting
        example_idx = []
        assert(len(self.tasks_to_complete) == 1)
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            for idx in range(obs.shape[0]):
                distance = np.linalg.norm(obs[idx][element_idx] - obs[idx][30+element_idx])
                complete = distance < BONUS_THRESH
                if complete:
                    example_idx.append(idx)
        print(len(example_idx))
        expert_obs = np.concatenate([dataset['observations'][t-1:t] for t in example_idx], axis=0)
        num_expert_obs = min(num_expert_obs, len(example_idx))
        indices = np.random.choice(
            len(expert_obs), size=num_expert_obs, replace=False)
        expert_obs = expert_obs[indices]
        return expert_obs

    def render(self, mode='human'):
        if mode =='rgb_array':
            camera = engine.MovableCamera(self.sim, 500, 500)
            camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            img = camera.render()
            return img
        else:
            super(KitchenTaskRelaxV1, self).render()


class KitchenMicrowaveKettleLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'slide cabinet']

class KitchenMicrowaveKettleBottomBurnerLightV0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'bottom burner', 'light switch']

class KitchenMicrowaveV0(KitchenBase):
    TASK_ELEMENTS = ['microwave']

class KitchenKettleV0(KitchenBase):
    TASK_ELEMENTS = ['kettle']

class KitchenSwitchV0(KitchenBase):
    TASK_ELEMENTS = ['light switch']

class KitchenCabinetV0(KitchenBase):
    TASK_ELEMENTS = ['slide cabinet']

class KitchenBurnerV0(KitchenBase):
    TASK_ELEMENTS = ['bottom burner']
