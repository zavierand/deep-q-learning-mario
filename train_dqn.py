'''
Training script for DQN agent on Mario Bros using Gymnasium and WandB for logging.
'''

import os

# import environment dependencies
import gymnasium as gym
import ale_py
from ale_py import ALEInterface

# import 
import torch

# import model
from dqn import DQN

# import plotting
import matplotlib.pyplot as plt

# import wandb for logging
import wandb

from dotenv import load_dotenv

load_dotenv()

print('CUDA Availability:', torch.cuda.is_available())

def __main__():
    API_KEY = os.getenv('API_KEY')
    if API_KEY is None:
        raise ValueError("API_KEY environment variable not set.")
    # instantiate the environment
    env = gym.make(
        'ALE/MarioBros-v5',
        obs_type = 'rgb',
        render_mode = 'rgb_array',
        frameskip = 4
    )

    # reset before training
    obs, info = env.reset()
    in_channels = env.render().shape[2]

    #env = ResizeObservation(env, shape=(84, 84))
    #env = FrameStack(env, num_stack=4)  # automatic frame stacking

    # wandb login
    wandb.login(key = API_KEY)

    # init wandb logging 
    wandb.init(
        entity="zavierand-new-york-university",
        project='dqn-mario',
        config={
            'learning_rate': 1e-3,
            'architecture': 'DQN',
            'epochs': 5000,
        },
    )

    epochs = 5000

    # instantiate the model
    dqn = DQN(env, in_channels, env.action_space.n)

    # train the model
    dqn._train(env, num_epochs = epochs)

    # env.close() 

if __name__ == "__main__":
    # run the main function
    __main__()