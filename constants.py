import pathlib
import numpy as np

### Task parameters
DATA_DIR = '/home/shixu/dev_shixu/act_humanoid/dataset_full_arm'
SIM_TASK_CONFIGS = {
    'pnp':{
        'dataset_dir': DATA_DIR + '/pnp',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['fixed']
    },
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
}

### Simulation envs fixed constants
DT = 0.02
START_ARM_POSE = [0, 0, 0, 0, -1.54, 0, 0, 0]
HAND_OPEN = [1, -0.8] + [0 for _ in range(8)]  # 10 DoF hand open -> 0
HAND_GRASP = [1.7, -0.5] + [0.7 for _ in range(8)]  # 10 DoF hand grasp, all fingers closed -> 1
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/dualarmhand0513/' # note: absolute path

def HAND_ACTION_NORMALIZE(hand_action):
    """
    Normalize a hand action (list of 10 joint values) to a single value in the range [0, 1].
    The single value is the average of the normalized values for all joints.
    """
    normalized_values = [
        (value - HAND_OPEN[i]) / (HAND_GRASP[i] - HAND_OPEN[i])
        for i, value in enumerate(hand_action)
    ]
    return sum(normalized_values) / len(normalized_values)

def HAND_ACTION_UNNORMALIZE(normalized_action):
    """
    Unnormalize a normalized hand action (a single value in [0, 1]) 
    back to the original joint value range for all 10 joints.
    """
    return np.array([
        HAND_OPEN[i] + normalized_action * (HAND_GRASP[i] - HAND_OPEN[i])
        for i in range(len(HAND_OPEN))
    ]).squeeze()


if __name__ == "__main__":
    pass
    # x = 1
    # HAND_ACTION = HAND_ACTION_UNNORMALIZE(x)
    # print(HAND_ACTION)

    # HAND_ACTION = [1.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # x = HAND_ACTION_NORMALIZE(HAND_ACTION)
    # print(x)