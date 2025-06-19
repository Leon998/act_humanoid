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
        print('Recording object trajectories')
        # Load action sequence from file
        joint_traj_path = os.path.join(dataset_dir, f"source/pnp_side_grasp_{episode_idx}/action_sequence.txt")
        joint_traj = np.loadtxt(joint_traj_path)[:,]
        # setup the environment
        env = make_sim_env()
        ts = env.reset()
        episode = [ts]
        object_trajectory = np.zeros((1, 3))
        # setup plotting
        # if onscreen_render:
        #     ax = plt.subplot()
        #     plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        #     plt.ion()
        for step in range(episode_len):
            action = joint_traj[step]
            ts = env.step(action)
            t_h2w, q_h2w = env.task.forward_kinematics('link7l', env.physics)
            t_o2w = env.task.get_inhand_obj_pos(t_h2w)
            object_trajectory = np.concatenate((object_trajectory, t_o2w.reshape(1, 3)), axis=0)
            episode.append(ts)
            # if onscreen_render:
            #     plt_img.set_data(ts.observation['images'][render_cam_name])
            #     plt.pause(0.001)
        # plt.close()
        # save the object trajectory
        object_trajectory = object_trajectory[1:]  # remove the initial zero row
        # Modify object_trajectory as per the requirement
        object_trajectory[:150] = object_trajectory[150]
        object_trajectory[250:] = object_trajectory[250]
        object_trajectory_path = os.path.join(dataset_dir, f"source/pnp_side_grasp_{episode_idx}/object_trajectory.txt")
        np.savetxt(object_trajectory_path, object_trajectory)

        # clear unused variables
        del env
        del episode

        # setup the environment
        print('Replaying')
        env = make_sim_env()

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = joint_traj[step]
            ts = env.step(action)
            t_o2w = object_trajectory[step]
            env.task.draw_traj('object_joint', t_o2w, env.physics)
            elbow_position, _ = env.task.forward_kinematics('link4l', env.physics)
            env.task.draw_traj('elbow_joint', elbow_position, env.physics)
            # print(ts.observation['qpos'][-1])  # check hand open-close status
            # print(ts.observation['env_state'])  # check object trajectory
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.001)
        plt.close()

        # save data
        """
        For each timestep:
        observations
        - qpos                  (9,)         'float64'
        - env_state             (3,)          'float64'

        action                  (9,)         'float64'
        """
        data_dict = {
            '/observations/qpos': [],
            '/observations/env_state': [],
            '/action': [],
        }
        joint_traj = joint_traj.tolist()
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/env_state'].append(ts.observation['env_state'])
            data_dict['/action'].append(action)

        # HDF5
        t0 = time.time()
        save_dir = dataset_dir + task_name + "/"
        dataset_path = os.path.join(save_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            qpos = obs.create_dataset('qpos', (max_timesteps, 9))
            env_state = obs.create_dataset('env_state', (max_timesteps, 3))
            action = root.create_dataset('action', (max_timesteps, 9))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {save_dir}')

        

if __name__ == '__main__':
    
    main()

