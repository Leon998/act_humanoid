import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from scipy.spatial.transform import Rotation as R

from constants import DT, XML_DIR, START_ARM_POSE, HAND_ACTION_UNNORMALIZE, HAND_ACTION_NORMALIZE

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name=None, box_position=None):
    xml_path = os.path.join(XML_DIR, f'fixed_robot.xml')
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = PnPTask(random=False, box_position=box_position)
    env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                              n_sub_steps=None, flat_observation=False)
    return env

class HumanoidTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        

    def before_step(self, action, physics):
        env_action = np.zeros(36)
        left_arm_action = action[:8]
        left_hand_action = HAND_ACTION_UNNORMALIZE(action[8:])
        env_action[18:26] = left_arm_action
        env_action[26:36] = left_hand_action
        super().before_step(env_action, physics)
        return
    
    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[18:26] = START_ARM_POSE

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    def forward_kinematics(self, link_name, physics):
        """Computes the forward kinematics of the left arm."""
        # Get the position and orientation of the left hand (link7l)
        position = physics.named.data.xpos[link_name]
        orientation = physics.named.data.xquat[link_name]

        return position, orientation
    
    # def draw_traj(self, name, position, physics):
    #     # Visualize the position in the simulation environment
    #     physics.named.data.qpos[name][:3] = position
    

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_arm_qpos = qpos_raw[18:26]
        left_hand_qpos_raw = qpos_raw[26:36]
        left_hand_qpos = [HAND_ACTION_NORMALIZE(left_hand_qpos_raw)]
        return np.concatenate([left_arm_qpos, left_hand_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[18:]
        left_arm_qvel = left_qvel_raw[:8]
        return np.array(left_arm_qvel)

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['fixed'] = physics.render(height=480, width=640, camera_id='fixed')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class PnPTask(HumanoidTask):
    def __init__(self, random=None, box_position=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.box_position = box_position

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # set box position
        box_pose = np.concatenate([np.array(self.box_position), np.array([1, 0, 0, 0])])
        box_start_idx = physics.model.name2id('box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], box_pose)
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        box_start_idx = physics.model.name2id('box_joint', 'joint')
        env_state = physics.data.qpos.copy()[box_start_idx : box_start_idx + 7]
        return env_state

    def get_reward(self, physics):
        reward = 1
        return reward


if __name__ == '__main__':
    pass

