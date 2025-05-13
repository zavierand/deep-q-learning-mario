# import wandb for logging
import wandb

# import os to get path for env
import os

# import copy to make deep copy for our target network
import copy

# import frameworks
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.utils as utils

# import conv layers, replay buffer, and policy
from cnn import ConvFeatureExtractor
from replay_buffer import ReplayBuffer, Transition
from policy import epsilon_greedy_policy

# store env vars in dotenv
from dotenv import load_dotenv

# import gym for fram stacking during training
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation

# import tqdm for progress bar
from tqdm import trange

# load environment variables
load_dotenv()

# model implementation
class DQN(nn.Module):
    def __init__(self, env, in_channels, num_actions):
        super(DQN, self).__init__()

        # set device to cuda if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f'Before training: CUDA available: {torch.cuda.is_available()}')
        
        self.conv_layers = ConvFeatureExtractor(in_channels)
        self.env = env

        # hyperparameters
        LR = 1e-3 # 0.001

        # preprocess
        '''self.transform = T.Compose([
            T.ToPILImage(),  # Convert numpy array to PIL Image
            T.Resize((84, 84)),  # resize image
            T.ToTensor(),  # Convert to tensor
        ])'''

        # dummy pass to determine flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 84, 84)   # original input from paper
            flattened_size = self.conv_layers(dummy_input).view(1, -1).size(1)

        # define model
        self.model = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),

            # one dense layer
            # nn.Linear(512, 512),
            # nn.ReLU(),

            # output layer
            nn.Linear(512, num_actions)
        )

        # create target network, for dqn stabilization
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

        # initialize optimizers
        self.criterion = nn.HuberLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # initialize buffer
        self.replay_buffer = ReplayBuffer(100000)

        # set model to cuda if available, otherwise cpu
        self.to(self.device)

    # preprocess image

    def preprocess(self, obs):
        frame = self.transform(obs)
        return frame  # Just return the single RGB frame
            
    def soft_update(self, tau):
        '''
        soft update of the target network's parameters.
        theta_target = tau * theta_online + (1 - tau) * theta_target
        '''
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def checkConvInputSize(self, obs):
        # check size of conv input to make sure enough frames are being passed
        # after preprocessing
        print(f'Shape of observation: {obs.shape}')
        obs = self.preprocess(obs).to(self.device)
        print(f'Shape of Conv Input: {obs.shape}')


    def forward(self, x):
        '''
        input: x = np array returned from env.render() method
        return: the q-value associated with this episode
        '''
        x = x.to(self.device)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.model(x)
    
    def _train(self, env, num_epochs=10000, batch_size=32, gamma=0.99):
        # initialize wandb
        '''wandb.init(project=os.getenv('WANDB_PROJECT'), entity=os.getenv('WANDB_LOGIN'))
        
        # log hyperparameters to wandb
        wandb.config.update({
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'gamma': gamma,
            'epsilon_decay': 0.999,
            'epsilon_min': 0.1
        })'''

        # epsilon values that will be used for training
        epsilon = 1.0
        epsilon_decay = 0.999
        epsilon_min = 0.05

        # keep track of rewards for avg calc
        rewards = np.array([])

        # begin training
        print(f'Training model for {num_epochs} epochs beginning.....')
        # training loop
        t = trange(num_epochs, desc="Training", leave=True)
        for epoch in t:
            # slight preprocessing
            obs, _ = env.reset()
            obs = obs / 255  # normalize the pixels between 0 and 1
            #obs = np.expand_dims(obs, axis = 0)
            obs = torch.as_tensor(obs, dtype = torch.float32).to(self.device)
            # print(obs.shape)    # light debugging
            done = False

            epoch_reward = 0.0
            step_count = 0
            while not done:
                # select action -> call to policy
                action = epsilon_greedy_policy(
                    env, 
                    self, 
                    obs.unsqueeze(0).float().to(self.device), 
                    epsilon
                )

                # step through environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                epoch_reward += reward
                step_count += 1
                done = terminated or truncated
                next_obs = next_obs / 255
                next_obs_tensor = torch.as_tensor(next_obs, dtype = torch.float32).to(self.device)

                # add reward to list
                reward = np.append(reward, epoch_reward)
                reward = np.clip(reward, -1, 1)

                # store transition in replay buffer
                self.replay_buffer.push(obs, action, next_obs_tensor, reward, done)
                obs = next_obs_tensor

                # train only if enough samples
                if len(self.replay_buffer) < batch_size:
                    continue

                # sample batch and convert to tensors
                transitions = self.replay_buffer.sample(batch_size)
                batch = Transition(*zip(*transitions))
                state_batch = torch.stack(batch.state).float().to(self.device)
                next_state_batch = torch.stack(batch.next_state).float().to(self.device)
                action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
                
                # convert batches to np arrays before assigning to tensor
                reward_batch = torch.tensor([r[0] for r in batch.reward], dtype=torch.float32).unsqueeze(1).to(self.device)
                done_batch_np = np.array(batch.done)
                done_batch = torch.as_tensor(done_batch_np, dtype=torch.long).unsqueeze(1).to(self.device)

                # print(f'State batch shape: {state_batch.shape}')
                # print(f'Action batch shape: {action_batch.shape}')

                # compute Q-values from current model
                q_values = self(state_batch).gather(1, action_batch)

                # compute target Q-values from target network
                with torch.no_grad():
                    next_state_conv = self.conv_layers(next_state_batch)
                    next_state_conv = next_state_conv.view(next_state_conv.size(0), -1)
                    max_next_q = self.target_model(next_state_conv).max(1)[0].unsqueeze(1)
                    target_q_values = reward_batch + gamma * max_next_q * (1 - done_batch)

                # compute loss and optimize
                loss = self.criterion(q_values, target_q_values)
                # print("Loss before backward:", loss.item())

                # backprop & adam step
                self.optimizer.zero_grad()
                loss.backward()

                # adding clipping to prevent explosing gradients
                #    - noted from previous training runs
                utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # update model
                self.optimizer.step()

            # decay epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # print and log epochs to ipynb and wandb
            t.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | Eps: {epsilon:.3f} | Reward: {epoch_reward:.1f}")

            # wandb log
            '''wandb.log({
                'epsilon': epsilon,
                'loss': loss.item(),
                'episode_reward': epoch_reward, 
                'steps': step_count,
                'avg_reward': np.mean(rewards[-100:])
            })'''

            # periodically update target network
            if (epoch + 1) % 10 == 0:
                self.soft_update(tau=0.001)  # You can adjust tau as needed

            # checkpoint model every 500 epochs
            if epoch % 500 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, f'./checkpoints/dqn_checkpoint_epoch_{epoch}.pt')

    # evaluate the model
    def _eval(self, env, num_episodes, render=False):
        self.eval()
        total_reward = 0.0
        rewards = []
        frames = []  # To store frames for animation

        for episode in range(num_episodes):
            do_render = render and (episode % 200 == 0)

            # reset environment and process the observation
            obs, _ = env.reset()
            obs /= 255
            obs = torch.as_tensor(obs, dtype = torch.float32).to(self.device)
            # obs = self.preprocess(obs).unsqueeze(0).float().to(self.device)
            done = False
            episode_reward = 0

            while not done:
                if do_render:
                    # capture frame when rendering is triggered
                    img = env.render()
                    frames.append(img)
                    wandb.log({"evaluation_frames": wandb.Image(img)})

                # Action selection
                action = epsilon_greedy_policy(
                    env,
                    self,
                    obs,
                    epsilon=0.0  # no exploration during evaluation
                )

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                next_obs = torch.as_tensor(next_obs, dtype = torch.float32).to(self.device)
                obs = next_obs
                # obs = self.preprocess(next_obs).unsqueeze(0).float().to(self.device)

            rewards.append(episode_reward)
            total_reward += episode_reward

            if (episode + 1) % 1000 == 0:
                print(f'Rendering agent at episode {episode + 1}')

        avg_reward = total_reward / num_episodes
        print(f'Average Reward per Episode: {avg_reward}')
        
        return avg_reward, rewards, frames
