'''
code for the epsilon greedy algorithm
'''

import numpy as np
import torch

def epsilon_greedy_policy(env, q_network, state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # explore
    else:
        with torch.no_grad():
            # Convert to tensor if not already
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)

            # Ensure correct shape: [batch_size, channels, height, width]
            if state.dim() == 3:  # [C, H, W]
                state = state.unsqueeze(0)  # -> [1, C, H, W]
            elif state.dim() != 4:
                raise ValueError(f"Unexpected state shape: {state.shape}")

            # Forward pass through the Q-network
            q_values = q_network(state)
            return torch.argmax(q_values, dim=1).item()  # exploit
