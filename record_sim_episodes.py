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


# def interpolate_action_sequence(action_sequence, episode_len):
#     """
#     Interpolate the action sequence to match the episode length.
#     :param action_sequence: np.ndarray, shape (n_steps, n_actions)
#     :param episode_len: int, desired length of the episode
#     :return: np.ndarray, interpolated action sequence
#     """
#     if len(action_sequence) < episode_len:
#         interpolated_actions = []
#         for col in range(action_sequence.shape[1]):
#             interpolated_col = np.interp(
#                 np.linspace(0, len(action_sequence) - 1, episode_len),
#                 np.arange(len(action_sequence)),
#                 action_sequence[:, col]
#             )
#             interpolated_actions.append(interpolated_col)
#         return np.array(interpolated_actions).T
#     else:
#         return action_sequence[:episode_len]
    
# def get_T_o2h(pose_path):
#     with open(pose_path, 'r') as file:
#         grasp_pose = file.readline().strip().split()
#         t_h2o = [float(grasp_pose[i]) for i in range(3)]  # 手相对于物体的平移量
#         q_h2o = [float(grasp_pose[i]) for i in range(3, 7)]  # 手相对于物体的旋转四元数, [x, y, z, w]
#     # Convert quaternion to rotation matrix
#     R_h2o = R.from_quat(q_h2o).as_matrix()
#     # Compute the inverse rotation matrix
#     R_o2h = R_h2o.T
#     # Compute the inverse translation
#     t_o2h = -R_o2h @ t_h2o
#     # Convert the inverse rotation matrix back to quaternion
#     q_o2h = R.from_matrix(R_o2h).as_quat()
#     return t_o2h, q_o2h

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
        # setup the environment
        env = make_sim_env()
        ts = env.reset()
        episode = [ts]
        object_trajectory = np.zeros((1, 3))
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = action_sequence[step]
            ts = env.step(action)
            t_h2w, q_h2w = env.task.forward_kinematics('link7l', env.physics)
            t_o2w = env.task.get_inhand_obj_pos(t_h2w)
            object_trajectory = np.concatenate((object_trajectory, t_o2w.reshape(1, 3)), axis=0)
            env.task.draw_traj(t_o2w, env.physics)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.001)
        plt.close()
        # save the object trajectory
        object_trajectory = object_trajectory[1:]  # remove the initial zero row
        # Modify object_trajectory as per the requirement
        object_trajectory[:150] = object_trajectory[150]
        object_trajectory[250:] = object_trajectory[250]
        object_trajectory_path = os.path.join(dataset_dir, f"source/pnp_side_grasp_{episode_idx}/object_trajectory.txt")
        np.savetxt(object_trajectory_path, object_trajectory)


if __name__ == '__main__':
    
    main()

