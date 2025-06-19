import pathlib

### Task parameters
DATA_DIR = '/home/shixu/dev_shixu/act_humanoid/dataset'
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
HAND_OPEN = [0 for _ in range(10)]  # 10 DoF hand open
HAND_GRASP = [1 for _ in range(10)]  # 10 DoF hand grasp, all fingers closed
HAND_ACTION_UNNORMALIZE = lambda x: HAND_GRASP if x else HAND_OPEN  # 0表示张开，为1表示闭合
HAND_ACTION_NORMALIZE = lambda x: 1 if x[0]>=0.5 else 0  # 0表示张开，为1表示闭合
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/dualarmhand0513/' # note: absolute path

