import numpy as np
import random
import math 
from collections import namedtuple
from itertools import count
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from Catan_Env.state_changer import state_changer
from replay_memory import ReplayMemory  

from Catan_Env.catan_env import Catan_Env

from Catan_Env.action_selection import action_selecter
from Catan_Env.random_action import random_assignment
from Catan_Env.game import Game
from Catan_Env.Interpreter import InterpretActions
from Configurations import *
from Artifacts import *

#different types of reward shaping: Immidiate rewards vps, immidiate rewards legal/illegal, immidiate rewards ressources produced, rewards at the end for winning/losing (+vps +legal/illegal)
class Log:
    def __init__(self):
        self.episodeRewardTracker =[]
        self.episodeRewardStep = []
        self.action_counts = [0] * TOTAL_ACTIONS
        self.random_action_counts = [0] * TOTAL_ACTIONS

    def clear(self):
        self.episodeRewardTracker =[]
        self.episodeRewardStep = []
        self.action_counts = [0] * TOTAL_ACTIONS
        self.random_action_counts = [0] * TOTAL_ACTIONS

class policy:
    def __init__(self, stochastic = False):
        self.stochastic = stochastic

    def get_action(self, predicted_model_q_values):
        if self.stochastic:
            action_probabilities = F.softmax(predicted_model_q_values, dim=1)
            action_distribution = Categorical(action_probabilities)
            action = action_distribution.sample().item()
        else:
            action = F.argmax(predicted_model_q_values)

        return action
    
    def get_expected_q_value(self,predicted_model_q_values):
        if self.stochastic:
            action_probabilities = F.softmax(predicted_model_q_values, dim=1)
            expected_q_value = torch.sum(action_probabilities * predicted_model_q_values, dim=1).detach()
        else:
            expected_q_value = predicted_model_q_values.max(1)[0].detach()
        return expected_q_value
    
    def get_legal_action(self, predicted_model_q_values, legal_actions):
        legal_indices = np.where(legal_actions == 1)[1]
        list_of_legal_q_values = [predicted_model_q_values[0][i].unsqueeze(0) for i in legal_indices]
        try:
            legal_actions_q_values = torch.cat(list_of_legal_q_values, dim=0)
        except:
            print("something went wrong creating the legal actions q values")
        if self.stochastic:
            legal_action_probabilities = F.softmax(legal_actions_q_values, dim=0)
            legal_action_distribution = Categorical(legal_action_probabilities)
            legal_policy_action = legal_indices[legal_action_distribution.sample().item()]
        else: 
            legal_policy_action = legal_indices[np.argmax(legal_actions_q_values)]
        return legal_policy_action

class DQNAgent:
    def __init__(self, model, device, memory_capacity=10000,
                 LR_START = .003, LR_END = .0002, LR_DECAY = 200000,
                 EPS_START = 1, EPS_END = .05, EPS_DECAY = 200000,
                 GAMMA = 0.999, BATCH_SIZE = 8):
        self.policy_net = model.to(device)
        self.target_net = model.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayMemory(memory_capacity)
        self.LR_START = LR_START
        self.LR_END = LR_END
        self.LR_DECAY = LR_DECAY
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR_START, amsgrad=True)
        self.steps_done = 0

class Catan_Training:
    def __init__(self, EPISODES_PER_LOOP, GAMES_PER_BENCHMARK, reward_function,device,model, stochastic_policy=False, memory=10000,
                 LR_START = .003, LR_END = .0002, LR_DECAY = 200000,
                 EPS_START = 1, EPS_END = .05, EPS_DECAY = 200000,
                 GAMMA = 0.999, BATCH_SIZE = 8, MIN_WIN_RATE = 0.25, LOG_FILE = None, CHECKPOINT = None):
        self.benchmark_games = GAMES_PER_BENCHMARK
        self.num_episodes_per_loop = EPISODES_PER_LOOP
        self.min_win_rate = MIN_WIN_RATE

        self.env = Catan_Env(reward_function)
        self.game = self.env.game
        self.device = device
        self.NEURAL_NET = model
        self.agent = DQNAgent(model, device, memory,
                                LR_START, LR_END, LR_DECAY,
                                EPS_START, EPS_END, EPS_DECAY,
                                GAMMA, BATCH_SIZE)
        self.policy = policy(stochastic=stochastic_policy)
        self.agent_policy_net = self.agent.policy_net
        self.agent_target_net = self.agent.target_net
        self.optimizer = self.agent.optimizer
        self.memory = self.agent.memory
        self.gamma = GAMMA
        self.cur_boardstate = None
        self.cur_vectorstate = None

        self.log = Log()
        self.log_file = LOG_FILE
        self.checkpoint = CHECKPOINT
        self.total_episodes = 0
        self.steps_done = self.agent.steps_done
        self.EpisodeData = []
        self.training = True
        self.train_test_change = False
        
        
    def add_epsiode_data(self,episode):
        if self.train_test_change:
            self.train_test_change = False
            self.EpisodeData = []
        if self.training:
            dict = {
                'Episode': episode,
                'player_0_win': int(self.game.winner==0),
                'game_length': self.env.phase.gamemoves +1,  #0 based  
                'player_0_knights': self.env.player0_log.average_knights_played[0],
                'player_0_roads': self.env.player0_log.average_roads_built[0],
                'player_0_settlements': self.env.player0_log.average_settlements_built[0],
                'player_0_cities': self.env.player0_log.average_cities_built[0],
                'player_0_dev_cards_bought': self.env.player0_log.average_development_cards_bought[0],
                'player_0_dev_cards_played': self.env.player0_log.average_development_cards_used[0],
                'player_0_dev_card_VP':self.env.player0.victorypoints_cards_new+self.env.player0.victorypoints_cards_new,
                'player_0_num_trades': self.env.player0_log.average_resources_traded[0],
                'player_1_knights': self.env.player1_log.average_knights_played[0],
                'player_1_roads': self.env.player1_log.average_roads_built[0],
                'player_1_settlements': self.env.player1_log.average_settlements_built[0],
                'player_1_cities': self.env.player1_log.average_cities_built[0],
                'player_1_dev_cards_bought': self.env.player1_log.average_development_cards_bought[0],
                'player_1_dev_cards_played': self.env.player1_log.average_development_cards_used[0],
                'player_1_dev_card_VP':self.env.player1.victorypoints_cards_new+self.env.player1.victorypoints_cards_new,
                'player_1_num_trades': self.env.player1_log.average_resources_traded[0],
                'player_0_victory_points': self.env.player0.victorypoints,
                'player_1_victory_points': self.env.player1.victorypoints,
                'average_model_loss': np.mean(self.game.average_q_value_loss),
                'Reward_over_episode':   sum(R*(self.gamma**T) for R,T in zip(self.log.episodeRewardTracker,self.log.episodeRewardStep))
                }
        else:
            dict = {
                'Episode': episode,
                'player_0_win': int(self.game.winner==0),
                'game_length': self.env.phase.gamemoves +1,  #0 based  
                'player_0_knights': self.env.player0_log.average_knights_played[0],
                'player_0_roads': self.env.player0_log.average_roads_built[0],
                'player_0_settlements': self.env.player0_log.average_settlements_built[0],
                'player_0_cities': self.env.player0_log.average_cities_built[0],
                'player_0_dev_cards_bought': self.env.player0_log.average_development_cards_bought[0],
                'player_0_dev_cards_played': self.env.player0_log.average_development_cards_used[0],
                'player_0_dev_card_VP':self.env.player0.victorypoints_cards_new+self.env.player0.victorypoints_cards_new,
                'player_0_num_trades': self.env.player0_log.average_resources_traded[0],
                'player_1_knights': self.env.player1_log.average_knights_played[0],
                'player_1_roads': self.env.player1_log.average_roads_built[0],
                'player_1_settlements': self.env.player1_log.average_settlements_built[0],
                'player_1_cities': self.env.player1_log.average_cities_built[0],
                'player_1_dev_cards_bought': self.env.player1_log.average_development_cards_bought[0],
                'player_1_dev_cards_played': self.env.player1_log.average_development_cards_used[0],
                'player_1_dev_card_VP':self.env.player1.victorypoints_cards_new+self.env.player1.victorypoints_cards_new,
                'player_1_num_trades': self.env.player1_log.average_resources_traded[0],
                'player_0_victory_points': self.env.player0.victorypoints,
                'player_1_victory_points': self.env.player1.victorypoints,
            }
        self.EpisodeData.append(dict)
        self.log.clear()



    def new_game(self):
        self.env.new_game()
        self.game = self.env.game
        self.cur_boardstate = state_changer(self.env)[0]
        self.cur_vectorstate = state_changer(self.env)[1]

    def select_action_using_policy(self, boardstate, vectorstate):
        sample = random.random()
        eps_threshold = self.agent.EPS_END + (self.agent.EPS_START - self.agent.EPS_END)*math.exp(-1. * self.steps_done / self.agent.EPS_DECAY)
        lr = self.agent.LR_END + (self.agent.LR_START - self.agent.LR_END) * math.exp(-1. * self.steps_done / self.agent.LR_DECAY)
        # Update the learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        if sample > eps_threshold:  # use epsilon greedy policy, select action from policy
            action_was_random = False
            with torch.no_grad():
                if self.game.cur_player == 0:
                    self.env.phase.actionstarted += 1
                    q_values = self.agent_policy_net(boardstate, vectorstate)
                    legal_actions = self.env.checklegalmoves()
                    policy_action = self.policy.get_legal_action(q_values, legal_actions)
                    if policy_action >= 4*11*21:
                        action_type = int(policy_action - 4*11*21 + 5)
                        position_y = 0
                        position_x = 0
                    else:
                        action_type = int(policy_action)//(11*21)+1
                        position_y = (int(policy_action) - ((action_type-1)*11*21))//21
                        position_x = int(policy_action) % 21 
                    action_selecter(self.env, action_type, position_x, position_y)
                    self.log.action_counts[policy_action] += 1
                    if self.env.phase.actionstarted >= 5:
                        action_selecter(self.env,5,0,0)
                
        else: # use epsilon greedy policy, select random action
            action_was_random = True
            action_type,position_x,position_y = random_assignment(self.env)
            if action_type > 4:
                policy_action = action_type + 4*11*21 - 5
            else:
                policy_action = (action_type-1)*11*21 + position_y*21 + position_x 
            self.log.random_action_counts[policy_action] += 1
            self.game.random_action_made = 1
        action_tensor = torch.tensor([[policy_action]], device=self.device, dtype=torch.long)
        return action_tensor, action_was_random
        

    def select_action_randomly(self):
        final_action,position_x,position_y = random_assignment(self.env)
        if final_action > 4:
            action = final_action + 4*11*21 - 5
        else:
            action = (final_action-1)*11*21 + position_y*21 + position_x 
        self.log.random_action_counts[action] += 1
        action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        self.game.random_action_made = 1
        return action

    def optimize_model(self):
        if len(self.memory) < self.agent.BATCH_SIZE:
            return
        Transition = namedtuple('Transition', ('cur_boardstate', 'cur_vectorstate', 'action', 'next_boardstate', 'next_vectorstate', 'reward'))
        transitions = self.memory.sample(self.agent.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s[0] is not None and s[1] is not None, zip(batch.next_boardstate, batch.next_vectorstate))), device=self.device, dtype=torch.bool)
        non_final_next_board_states = torch.cat([s for s in batch.next_boardstate if s is not None])
        non_final_next_vector_states = torch.cat([s for s in batch.next_vectorstate if s is not None])

        state_batch = (torch.cat(batch.cur_boardstate), torch.cat(batch.cur_vectorstate))
        action_batch = (torch.cat(batch.action))
        reward_batch = torch.cat(batch.reward)

        #the Q Values expected by current DQN model
        state_action_values = self.agent_policy_net(*state_batch).gather(1, action_batch) 

        # the Q values calculated using next state max Q value of all possible state action pairs (for all valid board and vector state combinations in the batch)
        # calculated by the DQN model and the actual reward received from current state-action pair
        next_state_expected_value = torch.zeros(self.agent.BATCH_SIZE, device=self.device)
        next_state_q_values = self.agent_target_net(non_final_next_board_states, non_final_next_vector_states)
        next_state_expected_value[non_final_mask] = self.policy.get_expected_q_value(next_state_q_values)
        expected_state_action_values = (next_state_expected_value * GAMMA) + reward_batch 

        #TO DO: Add hyper-parameter to select L1 vs MSE
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.game.average_q_value_loss.insert(0, loss.mean().item())
        while len(self.game.average_q_value_loss) > 1000:
            self.game.average_q_value_loss.pop(1000)
        
        self.game.average_reward_per_move.insert(0, self.env.phase.reward)
        while len(self.game.average_reward_per_move) > 1000:
            self.game.average_reward_per_move.pop(1000)

        self.game.average_expected_state_action_value.insert(0, expected_state_action_values.mean().item())
        while len(self.game.average_expected_state_action_value) > 1000:
            self.game.average_expected_state_action_value.pop(1000)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

    def simulate_match(self, PRINT_ACTIONS = False):
        self.env.new_game()
        game = self.game
        game.is_finished = 0
        done = False
        while not done:
            if game.cur_player == 0:
                with torch.no_grad():
                    self.env.phase.actionstarted += 1
                    cur_boardstate, cur_vectorstate = state_changer(self.env)
                    q_values = self.agent_policy_net(cur_boardstate, cur_vectorstate)
                    legal_actions = self.env.checklegalmoves()
                    policy_action = self.policy.get_legal_action(q_values, legal_actions)
                    can_end_turn = legal_actions[0][4*21*11]
                    if policy_action >= 4*11*21:
                        action_type = int(policy_action - 4*11*21 + 5)
                        position_y = 0
                        position_x = 0
                    else:
                        action_type = int(policy_action)//(11*21)+1
                        position_y = (int(policy_action) - ((action_type-1)*11*21))//21
                        position_x = int(policy_action) % 21 
                    if self.env.phase.actionstarted >= 5:
                        if can_end_turn: #ends turn after 5 policy actions
                            action_selecter(self.env,5,0,0)
                        else: #tries random actions if it can't end turn
                            # TO DO: evaluate trying to remove keep/discard resources from policy -> can we just randomly discard resources for both players when required. 
                            action_tensor = self.select_action_randomly()
                            random_action = True
                    else: 
                        action_selecter(self.env, action_type, position_x, position_y)
                        random_action = False
                        action_tensor = torch.tensor([[policy_action]], device=self.device, dtype=torch.long)
                if PRINT_ACTIONS:
                    InterpretActions(0,action_tensor, self.env, action_was_random=random_action, log_file=self.log_file)
            else:#Random AI takes turn
                self.env.phase.actionstarted = 0
                action = self.select_action_randomly()
                if PRINT_ACTIONS:
                    InterpretActions(1,action, self.env, action_was_random=True, log_file=self.log_file)
            
            self.env.phase.statechange = 0
            self.env.phase.reward = 0
            if game.is_finished == 1:
                done = True

        return game.winner

    def benchmark(self, PRINT_ACTIONS = False):
        # Load the network weights from a .pth file
        if self.checkpoint is not None:
            self.agent_policy_net.load_state_dict(torch.load(self.checkpoint, map_location=self.device, weights_only=True))
        print("Starting Benchmark")
        start_time = time.time()
        trained_policy_wins = 0
        random_policy_wins = 0
        for game_number in range(self.benchmark_games):
            self.log_file.write(f"\nStarting Game {game_number}")
            self.new_game()
            winner = self.simulate_match(PRINT_ACTIONS)
            self.add_epsiode_data(game_number)
            self.log.clear()
            if winner == 0:
                trained_policy_wins += 1
                self.log_file.write(f"\nTrained Policy Wins")
            elif winner == 1:
                random_policy_wins += 1
                self.log_file.write(f"\nRandom Policy Wins")
        trained_policy_win_rate = trained_policy_wins / self.benchmark_games
        random_policy_win_rate = random_policy_wins / self.benchmark_games
        print(f"benchmark finished in {time.time() - start_time} seconds")
        print(f"Trained Policy Win Rate: {trained_policy_win_rate * 100:.2f}%")
        print(f"Random Policy Win Rate: {random_policy_win_rate * 100:.2f}%")
        if trained_policy_win_rate < self.min_win_rate:
            return True
        return False


    def train(self, PRINT_ACTIONS = False):
        for i_episode in range(self.num_episodes_per_loop):
            self.total_episodes += 1   
            time_new_start = time.time()
            print(f"Starting Episode {self.total_episodes}")
            self.new_game()
            self.log_file.write(f"\n\n\nNew Game\n\n")
            for t in count():
                if self.game.cur_player == 1:
                    action = self.select_action_randomly() #player 1 uses random policy
                    if PRINT_ACTIONS:
                        InterpretActions(1,action, self.env, True, self.log_file)
                    self.env.phase.actionstarted = 0
                    if self.env.phase.statechange == 1:
                        if self.game.is_finished == 1:
                            self.game.cur_player = 0
                            self.cur_boardstate =  state_changer(self.env)[0]
                            self.cur_vectorstate = state_changer(self.env)[1]
                            cur_boardstate = self.cur_boardstate.clone().detach().unsqueeze(0).to(self.device).float()        
                            cur_vectorstate = self.cur_vectorstate.clone().detach().unsqueeze(0).to(self.device).float()
                            next_board_state, next_vector_state, reward, done = state_changer(self.env)[0], state_changer(self.env)[1], self.env.phase.reward, self.game.is_finished
                            ##add a sanity check on rewards. 
                            if reward != 0:
                                self.log.episodeRewardTracker.append(reward)
                                self.log.episodeRewardStep.append(t)
                            reward = torch.tensor([reward], device=self.device)
                            next_board_state = next_board_state.clone().detach().unsqueeze(0).to(self.device).float()
                            next_vector_state = next_vector_state.clone().detach().unsqueeze(0).to(self.device).float()
                            if done == 1:
                                self.env.phase.gamemoves = t
                                self.log_file.write(f"\ndone0")
                                next_board_state = None
                                next_vector_state = None
                            self.memory.push(cur_boardstate, cur_vectorstate,action,next_board_state, next_vector_state,reward)
                            self.cur_boardstate = next_board_state
                            self.cur_vectorstate = next_vector_state
                            self.optimize_model()
                            next_board_state = None
                            next_vector_state = None
                        if self.game.is_finished == 1:
                            self.env.phase.gamemoves = t
                            self.log_file.write(f"\ndone1")
                            self.game.is_finished = 0
                            break
                elif self.game.cur_player == 0:
                    self.cur_boardstate =  state_changer(self.env)[0]
                    self.cur_vectorstate = state_changer(self.env)[1]
                    cur_boardstate = self.cur_boardstate.clone().detach().unsqueeze(0).to(self.device).float()
                    cur_vectorstate = self.cur_vectorstate.clone().detach().unsqueeze(0).to(self.device).float()
                    action, action_was_random = self.select_action_using_policy(cur_boardstate, cur_vectorstate) #player 0 uses trained policy
                    if PRINT_ACTIONS:
                        InterpretActions(0, action, self.env, action_was_random, self.log_file)
                    if self.env.phase.statechange == 1:
                        next_board_state, next_vector_state, reward, done = state_changer(self.env)[0], state_changer(self.env)[1], self.env.phase.reward, self.game.is_finished
                        if reward != 0:
                            self.log.episodeRewardTracker.append(reward)
                            self.log.episodeRewardStep.append(t)
                        reward = torch.tensor([reward], device=self.device)
                        next_board_state = next_board_state.clone().detach().unsqueeze(0).to(self.device).float()
                        next_vector_state = next_vector_state.clone().detach().unsqueeze(0).to(self.device).float()
                        if done == 1:
                            self.env.phase.gamemoves = t
                            self.log_file.write(f"\ndone0")
                            next_board_state = None
                            next_vector_state = None
                        self.memory.push(cur_boardstate, cur_vectorstate,action,next_board_state, next_vector_state,reward)
                        self.cur_boardstate = next_board_state
                        self.cur_vectorstate = next_vector_state
                        self.optimize_model()

                        if done == 1:
                            self.env.phase.gamemoves = t
                            self.game.is_finished = 0
                            break
                    else:
                        sample = random.random()
                        if sample < 0.05:
                            next_board_state, next_vector_state, reward, done = state_changer(self.env)[0], state_changer(self.env)[1], self.env.phase.reward, self.game.is_finished
                            reward = torch.tensor([reward], device=self.device)
                            next_board_state = next_board_state.clone().detach().unsqueeze(0).to(self.device).float()
                            next_vector_state = next_vector_state.clone().detach().unsqueeze(0).to(self.device).float()
                            self.memory.push(cur_boardstate, cur_vectorstate,action,next_board_state, next_vector_state,reward)

                self.steps_done += self.env.phase.statechange
                self.env.phase.statechangecount += self.env.phase.statechange
                self.env.phase.statechange = 0
                self.game.random_action_made = 0
                self.env.phase.reward = 0
            self.add_epsiode_data(i_episode)
            self.game.average_time.insert(0, time.time() - time_new_start)
            if len(self.game.average_time) > 10:
                self.game.average_time.pop(10)
            self.game.average_moves.insert(0, t+1)
            if len(self.game.average_moves) > 10:
                self.game.average_moves.pop(10)
            if i_episode > 1 and t > 0:
                self.game.average_legal_moves_ratio.insert(0, (self.env.phase.statechangecount - statechangecountprevious)/t)
                if len(self.game.average_legal_moves_ratio) > 20:
                    self.game.average_legal_moves_ratio.pop(20)
            statechangecountprevious = self.env.phase.statechangecount
            self.env.phase.statechange = 0
            self.game.random_action_made = 0
            self.env.phase.reward = 0
            episode_time = time.time() - time_new_start
            print(f"Episode {self.total_episodes} finished in {episode_time} seconds")
            print(f'latest Optimizer Loss Avg: {np.mean(self.game.average_q_value_loss)}')

def main(TRAINING_LOOPS,EPISODES_PER_LOOP,GAMES_PER_BENCHMARK,MEMORY,MODEL_SELECT,REWARD_FUNCTION,STOCHASTIC,
         LR_START,LR_END,LR_DECAY,EPS_START,EPS_END,EPS_DECAY,GAMMA,BATCH_SIZE,PRINT_ACTIONS,MLFLOW_ADDRESS):
    
    param_dict = {
    "MEMORY": MEMORY,  # Size of the replay memory
    "MODEL_SELECT": MODEL_SELECT,  # Model selection (example: 'DQN_Big', 'DQN_Medium', 'DQN_Small')
    "REWARD_FUNCTION": REWARD_FUNCTION,  # Reward function used in the environment
    "TRAINING_LOOPS": TRAINING_LOOPS,  # Number of training loops
    "EPISODES_PER_LOOP": EPISODES_PER_LOOP, # Number of episodes per training loop
    "GAMES_PER_BENCHMARK": GAMES_PER_BENCHMARK, # Number of games for benchmarking winrate
    "LR_START": LR_START,  # Initial learning rate
    "LR_END": LR_END,  # Final learning rate
    "LR_DECAY": LR_DECAY,  # Decay factor for the learning rate
    "EPS_START": EPS_START,  # Initial epsilon for exploration
    "EPS_END": EPS_END,  # Final epsilon for exploration
    "EPS_DECAY": EPS_DECAY,  # Decay factor for epsilon
    "GAMMA": GAMMA,  # Discount factor
    "BATCH_SIZE": BATCH_SIZE  # Batch size for training
    }
    
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if MODEL_SELECT =='Small Pooling':
        from DQN.Neural_Networks.DQN_Small_w_pooling import DQN as dqn
    elif MODEL_SELECT =='Medium':
        from DQN.Neural_Networks.DQN_Medium import DQN as dqn
    else: 
        from DQN.Neural_Networks.DQN_Small import DQN as dqn
    model = dqn()
    model.to(device)
    LOG_FILE_NAME = f'Artifacts/log_{REWARD_FUNCTION}_{MODEL_SELECT}.txt'
    with open(LOG_FILE_NAME, 'w') as log_file:
        log_file.write(f"\nStart of log for {REWARD_FUNCTION} with {MODEL_SELECT} model\n\n")
        training = Catan_Training(EPISODES_PER_LOOP, GAMES_PER_BENCHMARK, REWARD_FUNCTION, device, model, stochastic_policy=STOCHASTIC, memory=MEMORY,
                              LR_START=LR_START, LR_END=LR_END, LR_DECAY=LR_DECAY,
                              EPS_START=EPS_START, EPS_END=EPS_END,
                              EPS_DECAY=EPS_DECAY, GAMMA=GAMMA, BATCH_SIZE=BATCH_SIZE, LOG_FILE=log_file)
        
        start_time = time.time()
  
        for _ in range(TRAINING_LOOPS):
            training.train(PRINT_ACTIONS)
            Experiment_name = f'{REWARD_FUNCTION}_{MODEL_SELECT}'
            PushArtifacts(Experiment_name,param_dict,training.agent_policy_net,training.EpisodeData,MLFLOW_ADDRESS,TrainingData=True,TestingData=False,TagID=0)
            should_break_training = training.benchmark(PRINT_ACTIONS)
            PushArtifacts(Experiment_name,param_dict,model,training.EpisodeData,MLFLOW_ADDRESS,TrainingData=False,TestingData=True,TagID=0)
            if should_break_training:
                print("Training stopped due to low win rate")
                break
            
        elapsed_time = time.time() - start_time
        print('Complete')
        print(f'steps over {training.total_episodes} episodes: {training.steps_done}')
        print(f'Elapsed time: {elapsed_time}')
        print(f'Optimizer steps: {len(training.game.average_q_value_loss)}')
        print(f'Optimizer Loss avg: {np.mean(training.game.average_q_value_loss)}')



main(TRAINING_LOOPS,EPISODES_PER_LOOP,GAMES_PER_BENCHMARK,MEMORY,MODEL_SELECT,REWARD_FUNCTION,STOCHASTIC,LR_START,LR_END,LR_DECAY,EPS_START,EPS_END,EPS_DECAY,GAMMA,BATCH_SIZE,PRINT_ACTIONS,MLFLOW_ADDRESS)

