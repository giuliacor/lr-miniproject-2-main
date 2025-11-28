"""
Run stable baselines 3 on quadruped env 
Check the documentation! https://stable-baselines3.readthedocs.io/en/master/
"""

import torch

# misc
import os
from datetime import datetime

# stable baselines 3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

# utils
from utils.utils import CheckpointCallback
from utils.file_utils import get_latest_model

# gym environment
from env.quadruped_gym_env import QuadrupedGymEnv

DEBUG_MODE = False

SEEDS = [0, 1, 2, 3, 4]

LEARNING_ALG = "PPO" # or "SAC"
LOAD_NN = False # if you want to initialize training with a previous model 
NUM_ENVS = 1    # how many pybullet environments to create for data collection
USE_GPU = True # make sure to install all necessary drivers 

# after implementing, you will want to test how well the agent learns with your MDP: 
# env_configs = {"motor_control_mode":"CPG",
#                "task_env": "FWD_LOCOMOTION", #  "LR_COURSE_TASK",
#                "observation_space_mode": "LR_COURSE_OBS"}
env_configs = {
    "motor_control_mode": "PD",
    "task_env": "FWD_LOCOMOTION",
    "observation_space_mode": "DEFAULT",
    "on_rack": False,
    "render": False,
    "record_video": False,
    "terrain": None
}

if USE_GPU and torch.cuda.is_available():
    gpu_arg = "auto" 
else:
    gpu_arg = "cpu"

if LOAD_NN:
    interm_dir = "./logs/intermediate_models/"
    log_dir = interm_dir + '' # add path
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    model_name = get_latest_model(log_dir)

# directory to save policies and normalization parameters
SAVE_PATH = './logs/intermediate_models/'+ datetime.now().strftime("%m%d%y%H%M%S") + '/'
os.makedirs(SAVE_PATH, exist_ok=True)

# checkpoint to save policy network periodically
checkpoint_callback = CheckpointCallback(save_freq=30000, save_path=SAVE_PATH,name_prefix='rl_model', verbose=2)

# create Vectorized gym environment
env = lambda: QuadrupedGymEnv(**env_configs)  
env = make_vec_env(env, monitor_dir=SAVE_PATH,n_envs=NUM_ENVS)

# normalize observations to stabilize learning (why?)
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

if LOAD_NN:
    env = lambda: QuadrupedGymEnv(**env_configs)
    env = make_vec_env(env, monitor_dir=SAVE_PATH, n_envs=NUM_ENVS)
    env = VecNormalize.load(stats_path, env)

# Multi-layer perceptron (MLP) policy of two layers of size _,_ each with tanh activation function
policy_kwargs = dict(net_arch=[256,256]) # act_fun=tf.nn.tanh

# What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
n_steps = 4096 
learning_rate = lambda f: 1e-4 
ppo_config = {  "gamma":0.99, 
                "n_steps": int(n_steps/NUM_ENVS), 
                "ent_coef":0.0, 
                "learning_rate":learning_rate, 
                "vf_coef":0.5,
                "max_grad_norm":0.5, 
                "gae_lambda":0.95, 
                "batch_size":128,
                "n_epochs":10, 
                "clip_range":0.2, 
                "clip_range_vf":1,
                "verbose":1, 
                "tensorboard_log":None, 
                "_init_setup_model":True, 
                "policy_kwargs":policy_kwargs,
                "device": gpu_arg}

# What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
sac_config={"learning_rate":1e-4,
            "buffer_size":300000,
            "batch_size":256,
            "ent_coef":'auto', 
            "gamma":0.99, 
            "tau":0.005,
            "train_freq":1, 
            "gradient_steps":1,
            "learning_starts": 10000,
            "verbose":1, 
            "tensorboard_log":None,
            "policy_kwargs": policy_kwargs,
            "seed":None, 
            "device": gpu_arg}

if LEARNING_ALG == "PPO":
    model = PPO('MlpPolicy', env, seed=SEEDS[3], **ppo_config)
elif LEARNING_ALG == "SAC":
    model = SAC('MlpPolicy', env, seed=SEEDS[3], **sac_config)
else:
    raise ValueError(LEARNING_ALG + 'not implemented')

if LOAD_NN:
    if LEARNING_ALG == "PPO":
        model = PPO.load(model_name, env)
    elif LEARNING_ALG == "SAC":
        model = SAC.load(model_name, env)
    print("\nLoaded model", model_name, "\n")

# Learn and save (may need to train for longer)
total_steps = 100000 if DEBUG_MODE else 1000000
model.learn(total_timesteps=total_steps, log_interval=1,callback=checkpoint_callback)

# Don't forget to save the VecNormalize statistics when saving the agent
model.save( os.path.join(SAVE_PATH, "rl_model" ) ) 
env.save(os.path.join(SAVE_PATH, "vec_normalize.pkl" )) 

if LEARNING_ALG == "SAC": # save replay buffer 
    model.save_replay_buffer(os.path.join(SAVE_PATH,"off_policy_replay_buffer"))