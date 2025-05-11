# import environment dependencies
import gymnasium as gym
import ale_py
from ale_py import ALEInterface

# import saved model
import torch

# import model
from dqn import DQN

# import plotting
import matplotlib.pyplot as plt

# import wandb for logging
import wandb

# import model
from dqn import DQN

def main():
    env = gym.make(
        'ALE/MarioBros-v5',
        obs_type = 'rgb',
        render_mode = 'rgb_array',
        frameskip = 5
    )

    # instantiate the model
    obs, info = env.reset()
    in_channels = env.render().shape[2]

    wandb.login()

    # init wandb logging 
    wandb.init(
        entity="zavierand-new-york-university",
        project='dqn-mario',
        config={
            'learning_rate': 1e-2,
            'architecture': 'DQN',
            'epochs': 5000,
        },
    )

    epochs = 5000

    # instantiate the model
    dqn = DQN(env, in_channels, env.action_space.n)

    # train the model
    dqn._train(env, num_epochs = epochs)

main()