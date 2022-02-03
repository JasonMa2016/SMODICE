""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel
import numpy as np
import random


WALL = 10
EMPTY = 11
GOAL = 12


def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr


def point_maze(maze_str, goal):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel('point_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(type="2d",name="groundplane",builtin="checker",rgb1="0.2 0.3 0.4",rgb2="0.1 0.2 0.3",width=100,height=100)
    asset.texture(name="skybox",type="skybox",builtin="gradient",rgb1=".4 .6 .8",rgb2="0 0 0",
               width="800",height="800",mark="random",markrgb="1 1 1")
    asset.material(name="groundplane",texture="groundplane",texrepeat="20 20")
    asset.material(name="wall",rgba=".7 .5 .3 1")
    asset.material(name="target",rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4",diffuse=".8 .8 .8",specular="0.1 0.1 0.1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name='ground',size="40 40 0.25",pos="0 0 -0.1",type="plane",contype=1,conaffinity=0,material="groundplane")

    particle = worldbody.body(name='particle', pos=[1.2,1.2,0])
    particle.geom(name='particle_geom', type='sphere', size=0.1, rgba='0.0 0.0 1.0 0.0', contype=1)
    particle.site(name='particle_site', pos=[0.0,0.0,0], size=0.2, rgba='0.3 0.6 0.3 1')
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    # hack for visualization for maze2d-example-v1
    worldbody.site(name='target_site', pos=[goal[0]+1,goal[1]+1,0], size=0.2, material='target')

    width, height = maze_arr.shape
    scaling = 4
    for w in range(width):
        for h in range(height):
            if maze_arr[w,h] == WALL:
                worldbody.geom(conaffinity=1,
                               type='box',
                               name='wall_%d_%d'%(w,h),
                               material='wall',
                               pos=[w+1.0,h+1.0,0],
                               size=[0.5,0.5,0.2])

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel

EXAMPLE_MAZE = \
        "#########\\"+\
        "#OOOGOOO#\\"+\
        "#OOOOOOO#\\"+\
        "#OOOOOOO#\\"+\
        "#GOOOOOG#\\"+\
        "#OOOOOOO#\\"+\
        "#OOOOOOO#\\"+\
        "#OOOGOOO#\\"+\
        "#########"

LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"


class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(self,
                 maze_spec=U_MAZE,
                 reward_type='sparse',
                 reset_target=False,
                 init_target=None,
                 **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)

        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations.sort()

        if init_target is None:
            self._target = np.array([1.0,4.0])
        else:
            self._target = np.array(init_target)
        self._max_episode_steps = 200

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        self.goal_id = 0
        
        model = point_maze(maze_spec, self._target)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1)
        utils.EzPickle.__init__(self)

        # if len(self.goal_locations) == 1:
        #     if init_target is None:
        #         self.set_target(self.goal_locations[0])
        # elif len(self.goal_locations) > 1:
        #     self.set_target(self.goal_locations[self.goal_id])
            # self.goal_id = (self.goal_id + 1) % len(self.goal_locations)
            # raise ValueError("More than 1 goal specified!")
        # else:
            # If no goal, use the first empty tile
        #     self.set_target(np.array(self.reset_locations[0]).astype(self.observation_space.dtype))
        # self.empty_and_goal_locations = self.reset_locations + self.goal_locations

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        # self.set_marker()
        ob = self._get_obs()
        done = False
        if self.reward_type == 'sparse':
            # print(np.linalg.norm(ob[0:2] - self._target))
            if np.linalg.norm(ob[0:2] - self._target) <= 0.17:
                reward = 1.0
                done = True 
            else:
                # terminate when any goal is reached
                for goal in self.goal_locations:
                    # print(np.linalg.norm(ob[0:2] - goal), goal)
                    if np.linalg.norm(ob[0:2] - goal) <= 0.17:
                        done = True 
                reward = 0.0          
        elif self.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(ob[0:2] - self._target))
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        target_location = self.goal_locations[self.goal_id]
        target_location = target_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        self.goal_id = (self.goal_id + 1) % len(self.goal_locations)
        # if target_location is None:
        #     idx = self.np_random.choice(len(self.empty_and_goal_locations))
        #     reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        #     target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        self._target = target_location
    def set_marker(self):
        self.data.site_xpos[self.model.site_name2id('target_site')] = np.array([self._target[0], self._target[1], 0.0])

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        # idx = self.np_random.choice(len(self.empty_and_goal_locations))
        # reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        # qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        # qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        # self.set_state(qpos, qvel)

        # example-maze
        if self.str_maze_spec == EXAMPLE_MAZE:
            self.reset_to_location((4,4))
        # umaze
        elif self.str_maze_spec == U_MAZE:
            self.reset_to_location((3,1))
        else:
            raise NotImplementedError
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.elevation = -90 
        # pass

