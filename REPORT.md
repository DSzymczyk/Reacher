# Report
Learning algorithm used to solve environment is DDPG - Deep Deterministic Policy Gradient.

In DDPG the agent collects experiences  and stores these experience samples into a replay buffer. 
On every n steps, the agent pulls out a batch from the replay buffer that is sampled at random. 
The agent then uses this batch to calculate a bootstrapped TD target and train a Q-function. 

In DDPG we use deterministic policies to approximate greedy action. We directly approximate the best action 
in the next state using a policy function, then we use that action with the Q-function to get the max value.

In DDPG we use off-policy exploration strategy. In this project Ornstein-Uhlenbeck Process was used as exploration 
strategy.


### Adjustable hyper parameters in algorithm contains:  
  - `policy learning rate` - learning rate for optimizer, determines step while minimizing loss function.
  - `value learning rate` - learning rate for optimizer, determines step while minimizing loss function.
  - `weight decay` - amount of penalty added to the cost which can lower model weights.
  - `gamma` - discount value for rewards. Determines how much past rewards are valued. Used for calculating q value.
  - `tau` - used for updating target model. Determines update rate from online model.
  - `buffer size` - maximum size of replay buffer. Determines how many experiences are stored at once.
  - `batch size` - size of batch used for training. Determines size of experiences batch used for learning.
  - `learn_every` - steps interval. Learning will occur every `learn_every` steps.


Algorithm has the highest performance when hyper parameters are set to:
  - `policy learning rate`: 0.0001 
  - `value learning rate`: 0.001 
  - `weight decay`: 0
  - `gamma`: 0.99
  - `tau`: 0.001
  - `buffer size`:1000000
  - `batch size`: 512
  - `learn_every`: 4

### Ornstein Uhlenbeck Process Parameters
  - `mu` - represents a long-term mean of the OU process
  - `theta` - mean reverting speed value
  - `sigma` - deviation of stochastic factor

Algorithm has the highest performance when hyper parameters are set to:
  - `mu`: 0.0
  - `theta`: 0.001
  - `sigma`: 0.002

Agent with such hyper parameters learns very fast and reach 30 points first time in 36 episodes. Environment was solved 
in 3 episodes.

### Training Results
![Alt Text](report/training_results.png)

### Ideas for future work
 - Solve environment using PPO or TD3
 - Solve Crawl environment using DDPG