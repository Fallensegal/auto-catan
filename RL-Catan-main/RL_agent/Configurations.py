PRINT_ACTIONS = True

# Starting learning rate for the optimizer
LR_START = 0.0000003

# Ending learning rate for the optimizer
LR_END = 0.0000001

# Decay rate for the learning rate
LR_DECAY = 2000000

# Starting value for the exploration rate (epsilon) in the epsilon-greedy policy
EPS_START = 1

# Ending value for the exploration rate (epsilon) in the epsilon-greedy policy
EPS_END = 0.05

# Decay rate for the exploration rate (epsilon) in the epsilon-greedy policy
EPS_DECAY = 200000

# Discount factor for future rewards in the Q-learning algorithm
GAMMA = 0.99

# Batch size for training the DQN model
BATCH_SIZE = 64

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

MODEL_SELECT = 'Small'
'''
        There are 3 model sizes: 
        1. 'Small'
        2. 'Medium'
        3. 'Large'
'''

STOCHASTIC = False #true if you want to use a stochastic policy instead of a deterministic one

#number of episodes
NUM_EPISODES = 10

MEMORY = 1000

# Total number of possible actions in the environment
TOTAL_ACTIONS = 21*11*4 + 41

#I might do a mix later on
#target_net_state_dict = target_net.state_dict()
#policy_net_state_dict = agent1_policy_net.state_dict()
#for key in policy_net_state_dict:
#    target_net_state_dict[key] = TAU*policy_net_state_dict[key] + (1-TAU)*target_net_state_dict[key]
#target_net.load_state_dict(target_net_state_dict)
# Soft update coefficient for updating the target network in the DQN algorithm
#TAU = 0.002