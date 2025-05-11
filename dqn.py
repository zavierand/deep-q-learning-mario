# import wandb for logging
import wandb

# import os to get path for env
import os

# import copy to make deep copy for our target network
import copy

# import frameworks
import torch
from torch import nn
import torch.optim as optim
import torchvision.transforms as T

# import conv layers, replay buffer, and policy
from cnn import ConvFeatureExtractor
from replay_buffer import ReplayBuffer, Transition
from policy import epsilon_greedy_policy

# store env vars in dotenv
from dotenv import load_dotenv

load_dotenv()

# model implementation
class DQN(nn.Module):
    def __init__(self, env, in_channels, num_actions):
        super(DQN, self).__init__()

        # call cuda, speed up training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.conv_layers = ConvFeatureExtractor(in_channels)
        self.env = env

        # hyperparameters
        LR = 1e-2

        # preprocess
        self.transform = T.Compose([
            T.ToPILImage(),  # Convert numpy array to PIL Image
            T.Resize((84, 84)),  # resize image
            T.ToTensor(),  # Convert to tensor
        ])

        # dummy pass to determine flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 84, 84)
            flattened_size = self.conv_layers(dummy_input).view(1, -1).size(1)

        # define model
        self.model = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),

            # one dense layer
            nn.Linear(512, 512),
            nn.ReLU(),

            # output layer
            nn.Linear(512, num_actions)
        )

        # create target network, for dqn stabilization
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

        # initialize optimizers
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # initialize buffer
        self.replay_buffer = ReplayBuffer(10000)

    # preprocess image
    def preprocess(self, obs):
        '''
        Preprocess the input image from the environment.
        This resizes the image and converts it to a tensor.
        '''
        return self.transform(obs)

    def forward(self, x):
        '''
        input: x = np array returned from env.render() method
        return: the q-value associated with this episode
        '''

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.model(x)
    
    def _train(self, env, num_epochs=100, batch_size=32, gamma=0.99):
        # initialize wandb
        wandb.init(project=os.getenv('WANDB_PROJECT'), entity=os.getenv('WANDB_LOGIN'))
        
        # log hyperparameters to wandb
        wandb.config.update({
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'gamma': gamma,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.1
        })

        # epsilon values that will be used for training
        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.1

        # print bar for notebook logging
        bar_width = 30

        # training loop
        for epoch in range(num_epochs):
            obs, _ = env.reset()
            obs = self.preprocess(obs)
            done = False

            while not done:
                # select action -> call to policy
                action = epsilon_greedy_policy(
                    env, 
                    self, 
                    obs.unsqueeze(0).float(), 
                    epsilon
                )

                # step through environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_obs_tensor = self.preprocess(next_obs)

                # store transition in replay buffer
                self.replay_buffer.push(obs, action, next_obs_tensor, reward, done)
                obs = next_obs_tensor

                # train only if enough samples
                if len(self.replay_buffer) < batch_size:
                    continue

                # sample batch and convert to tensors
                transitions = self.replay_buffer.sample(batch_size)
                batch = Transition(*zip(*transitions))

                state_batch = torch.stack(batch.state).float()
                next_state_batch = torch.stack(batch.next_state).float()
                action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
                done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

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

                # compute loss and optimization
                loss = self.criterion(q_values, target_q_values)
                # print("Loss before backward:", loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # decay epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # print training progress
            progress = int(bar_width * (epoch + 1) / num_epochs)
            bar = "#" * progress + "-" * (bar_width - progress)

            # print epochs
            print(f"Epoch {epoch+1}/{num_epochs}, [{bar}] Loss: {loss.item():.4f}, Epsilon: {epsilon:.4f}")

            # periodically update target network
            if (epoch + 1) % 5 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        torch.save(self.model.state_dict(), "dqn_trained.pt")


    # evaluate the model
    def _eval(self, env, num_episodes, render=False):
        self.eval()
        total_reward = 0.0
        rewards = []
        frames = []  # To store frames for animation

        for episode in range(num_episodes):
            do_render = render and (episode % 200 == 0)

            # Reset environment and process the observation
            obs, _ = env.reset()
            obs = self.preprocess(obs).unsqueeze(0).float()
            done = False
            episode_reward = 0

            while not done:
                if do_render:
                    # Capture frame when rendering is triggered
                    img = env.render()
                    frames.append(img)

                # Action selection
                action = epsilon_greedy_policy(
                    env,
                    self,
                    obs,
                    epsilon=0.0  # No exploration during evaluation
                )

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                obs = self.preprocess(next_obs).unsqueeze(0).float()

            rewards.append(episode_reward)
            total_reward += episode_reward

            if (episode + 1) % 1000 == 0:
                print(f'Rendering agent at episode {episode + 1}')

        avg_reward = total_reward / num_episodes
        print(f'Average Reward per Episode: {avg_reward}')
        
        return avg_reward, rewards, frames
