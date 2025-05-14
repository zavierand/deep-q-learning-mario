'''
Training script for DQN agent on Mario Bros using Gymnasium and WandB for logging.
'''

import os

# import environment dependencies
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
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
        obs_type = 'grayscale',
        render_mode = 'rgb_array',
        frameskip = 4
    )

    # preprocess env
    env = ResizeObservation(env, (84, 84))
    # env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)

    # reset before training
    obs, info = env.reset()

    # slight preprocessing before being passed into training
    #obs = obs.transpose(0, 3, 1, 2).reshape(210, 160, 12)
    # print(obs.shape)

    in_channels = obs.shape[0]
    # print(f'In-channels: {in_channels}')

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

    epochs = 10000

    # instantiate the model
    dqn = DQN(env, in_channels, env.action_space.n)

    # train the model
    dqn._train(
        env, 
        num_epochs = epochs, 
        resume = True,
        checkpoint_path = './checkpoints/train_02/dqn_checkpoint_epoch_4500.pt'
    )

    env.close() 

if __name__ == "__main__":
    # run the main function
    __main__()