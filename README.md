# Reinforcement Learning using Deep-Q-Networks on Mario
This README will be updated as the project is built out.

# Overview
The goal of this project is to reimplement the [existing paper](https://arxiv.org/abs/1312.5602) on deep-q-networks for reinforcement learning by [DeepMind](https://deepmind.google) with a slight variation through what was taught during Spring 2025 CSCI-UA 473: Fundamentals of Machine Learning with Pascal Wallisch. Mario was the chosen video game between these because for my parents growing up, they would always visit the arcade and play either Pacman or Donkey Kong. Eventually, my mom moved over to Mario and I grew up playing a lot of Mario Bros. games on the GameBoy and Nintendo DS. I also have internal beef with Donkey Kong, so I decided to roll with Mario - given as well, the discrete spaces for these two games were very similar with both having 18 actions that can be quantified into a discrete space.

# Q-Learning
To understand q-learning, we need to understand the different type of reinforcement learning algorithms. With reinforcement learning, there are two categories of algorithms - value-based algorithms and policy-based algorithms. Now, we must understand the state-space of the given problem. With value-based algorithms, we are trying to observe how an RL agent will react given a set of constraints. This set of constraints placed on the agent will essentially produce a value, the max reward the policy can achieve while undergoing said policy. Essentially, the agent is trying to learn the best possible set of states to visit and actions to take to maximize the most possible reward. The q-value, is the optimal expected reward from taking an action, $a$, in state, $s$, and following that policy. The function that yields a q-value is called a "q-value function".

## Q-Value and the Bellman Equation
Now, we can start with understanding how to compute the q-value by introducing a q-table. A q-table is a matrix that we can use to store values q-values of each state-action pair. We can think back to simple markov decision processes, where  To compute a q-value, we can solve the Bellman equation:

$$Q^{*}(s_t, a_t) \Leftarrow Q(s, a) + \alpha(r + \gamma \text{max}Q(s', a') - Q(s, a))$$

where solving for $Q^{*}(s_t, a_t)$ provides us the q-value. The Bellman Equation takes in two arguments to the function, a state, $s_t$, at time, $t$, and an action $a_t$ at time $t$.To understand each term in the equation, we can break it down where:
    - $\alpha$ is the learning rate,
    - $r$ is the immediate reward at state, $s$, after taking action $a$
    - $\gamma$ is the discount factor - how much we value future rewards
    - $s'$ is the next state

We now have a function that we want to our agent to best estimate to find the optimal set of states to visit and actions to take to obtain the maximal q-value. However, we still want to
[*to be updated*]

# Deep-Q-Networks
To implement the deeq-q-network, I followed something similar to what was implemented by the team at DeepMind, stated in their paper. My model consists of a convolutional head that processes, filters, and flattens the current image before being passed into a fully connected network consisting of just 2 hidden layers, each with 512 nodes.

## Epsilon-Greedy Algorithm
To figure out the best action the model should take during training and evaluating, we use the epsilon-greedy algorithm. Simply put, we initialize epsilon, $\epsilon$, our decay factor, and the minimum possibe epsilon value. As the agent explores, it will select actions uniformly at random from the action space. In my (code)[./policy.py], we can see that with the following block:
```
if np.random.rand() < epsilon:
        return env.action_space.sample()  # explore
```
otherwise, the agent will just continue exploiting the sequence of actions that it has taken already:

```
# in the code

q_values = q_network(state)
            return torch.argmax(q_values, dim=1).item()  # exploit
```

Mathematically, we can represent the epsilon-greedy algorithm as the following function:

$$\alpha = \text{argmax}Q'(a, a')$$ 

or as the DeepMind paper writes:

$$a_t = \text{max}_aQ*(\phi(s_t), a_t; \theta)$$

[*to be updated*]
## Replay Buffer
The replay buffer acts as the experience replay for the q-network to learn from. The buffer acts as data collection, storing experiences as a sequence of states, actions, next states,  and rewards, and whether the episode ended or not. During data collection, these "experiences" are stored in the replay buffer, where we sample a random sequence of actions to pick next, and the model will take that action as the next possible action in the current epoch. Basically, it allows us to learn from the entire discrete space as oppose to actions closely related - which if we think back to earlier in the semester where we talked about collinearity, is something we want to avoid as best as possible.

[*to be updated*]