import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim，只是为了初始化

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        # DataLoader通过此函数迭代调用数据集中的数据，每次调用都按照下面的规则对数据做一些调整
        sample_full_episode = True # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)  # 猜测：采样随机的时刻作为起始时刻，增加数据集的多样性，即从任意时刻开始都能做完任务，从而提高鲁棒？
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            env_state = root['/observations/env_state'][start_ts]
            # image_dict = dict()
            # for cam_name in self.camera_names:
            #     image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        # all_cam_images = []
        # for cam_name in self.camera_names:
        #     all_cam_images.append(image_dict[cam_name])
        # all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        # image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        env_state_data = torch.from_numpy(env_state).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        # image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        # image_data = image_data / 255.0
        # env_state_data[:3] = (env_state_data[:3] - self.norm_stats["env_pos_mean"]) / self.norm_stats["env_pos_std"]
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return env_state_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_env_pos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            env_pos = root['/observations/env_state'][()][:,:3]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_env_pos_data.append(torch.from_numpy(env_pos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_env_pos_data = torch.stack(all_env_pos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    # normalize env_pos data
    env_pos_mean = all_env_pos_data.mean(dim=[0, 1], keepdim=True)
    env_pos_std = all_env_pos_data.std(dim=[0, 1], keepdim=True)
    env_pos_std = torch.clip(env_pos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "env_pos_mean": env_pos_mean.numpy().squeeze(), "env_pos_std": env_pos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]  # 对应EpisodicDataset中的episode_ids
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

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