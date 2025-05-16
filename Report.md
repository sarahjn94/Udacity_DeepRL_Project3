# Udacity_DeepRL_Project3

# Project 3: Collaboration and Competition

### Learning Algorithm

For this Collaboration and Competition project, the implementation used a multi-agent deep deterministic policy gradient (MADDPG) learning algorithm.
	
 	- Actor-Critic architecture
		- Both had local and target network with soft updates on target network
		- Input layer of size 24
		- Actor output layer of size 2
		- Critic output layer of size 1
		- First hidden layer of size 256 with ReLU activation and batch normalization
		- Second hidden layer of size 128 with ReLU activation
	- Used a shared replay buffer to store past experiences for collaborative learning between agents
	- Each agent has their own actor-critic network
	- Soft target network updates
	- Gradient clipping added
	- Ornstein-Uhlenbeck noise process with noise decay to explore action space

	- Hyperparameters:
		- Replay buffer size: 1e6
		- Batch size: 128
		- Gamma (discount factor): 0.99
		- Tau: 1e-3
		- LR_Actor: 1e-5
		- LR_Critic: 1e-4
		- Weight decay: 0
		- Training episodes: 3000
		- Noise:
			- theta: 0.15
			- sigma: 0.3
			- decay_rate: 0.998

### Plot of Rewards

![Graph](/outputGraph.png)

Environment solved in 2331 episodes!	Average Score: 0.51


### Ideas for Future Work

	- Try other noise parameter configurations and options
	- Deeper NN architecture
	- Implement prioritized experience replay for more efficiency
	- More hyperparameter tuning
	- Try other learning algorithms or combinations
