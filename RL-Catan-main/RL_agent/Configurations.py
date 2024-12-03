PRINT_ACTIONS = False

# Batch size for training the DQN model
BATCH_SIZE = 8

# Discount factor for future rewards in the Q-learning algorithm
GAMMA = 0.999

# Starting value for the exploration rate (epsilon) in the epsilon-greedy policy
EPS_START = 1

# Ending value for the exploration rate (epsilon) in the epsilon-greedy policy
EPS_END = 0.05

# Decay rate for the exploration rate (epsilon) in the epsilon-greedy policy
EPS_DECAY = 200000

# Soft update coefficient for updating the target network in the DQN algorithm
TAU = 0.002

# Starting learning rate for the optimizer
LR_START = 0.003

# Ending learning rate for the optimizer
LR_END = 0.0002

# Decay rate for the learning rate
LR_DECAY = 2000000

# Total number of possible actions in the environment
TOTAL_ACTIONS = 21*11*4 + 41

#Reward Functions
REWARD_FUNCTION = 'Incremental_VP'
'''
        There are several Reward Functions that can be called: 
        1. 'Basic'
        2. 'Differential_VP'
        3. 'Incremental_VP'
        4. 'Large_Magnitude'
        5. 'High_Sparsity_VP'
'''