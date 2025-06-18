import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
from scipy.spatial.transform import Rotation as R

from constants import SIM_TASK_CONFIGS, START_ARM_POSE
from sim_env import make_sim_env, BOX_POSE

import IPython
e = IPython.embed


def main():
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name ="pnp"
    dataset_dir = "dataset/"
    num_episodes = 3
    onscreen_render = True
    render_cam_name = 'fixed'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']

    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        # Load action sequence from file
        action_sequence_path = os.path.join(dataset_dir, f"source/pnp_side_grasp_{episode_idx}/action_sequence.txt")
        action_sequence = np.loadtxt(action_sequence_path)[:,]
        object_trajectory_path = os.path.join(dataset_dir, f"source/pnp_side_grasp_{episode_idx}/object_trajectory.txt")
        object_trajectory = np.loadtxt(object_trajectory_path)[:,]
        # setup the environment
        env = make_sim_env()
        ts = env.reset()
        episode = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = action_sequence[step]
            ts = env.step(action)
            t_o2w = object_trajectory[step]
            env.task.draw_traj(t_o2w, env.physics)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.001)
        plt.close()


if __name__ == '__main__':
    
    main()

