import numpy as np
import random
import math 
from collections import namedtuple, deque
from itertools import count
import time
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, project_root)

#from DQN.config import *
#from DQN.log import *


from Catan_Env.state_changer import state_changer
from DQN.replay_memory import ReplayMemory  

from Catan_Env.catan_env import Catan_Env

from Catan_Env.action_selection import action_selecter
from Catan_Env.random_action import random_assignment
from Catan_Env.game import Game
#from RL_agent.DQN.Neural_Networks.DQN_Small import DQN as dqn

from Configurations import *
from Catan_Env.Interpreter import InterpretActions
#different types of reward shaping: Immidiate rewards vps, immidiate rewards legal/illegal, immidiate rewards ressources produced, rewards at the end for winning/losing (+vps +legal/illegal)
class Log:
    def __init__(self):
        self.average_victory_points = []
        self.average_resources_found = []
        self.final_board_state = 0 
        self.AI_function_calls = 0 
        self.successful_AI_function_calls = 0
        self.average_development_cards_bought = []
        self.average_roads_built = []
        self.average_settlements_built = []
        self.average_cities_built = []
        self.average_knights_played = []
        self.average_development_cards_used = []
        self.average_resources_traded = []
        self.average_longest_road = []

        self.total_resources_found = 0
        self.total_development_cards_bought = 0
        self.total_roads_built = 0
        self.total_settlements_built = 0
        self.total_cities_built = 0
        self.total_development_cards_used = 0
        self.total_resources_traded = 0
        self.total_knights_played = 0

        self.steps_done = 0

        self.action_counts = [0] * TOTAL_ACTIONS
        self.random_action_counts = [0] * TOTAL_ACTIONS
        self.episode_durations = []

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
    def __init__(self, reward_function,device,model, num_episodes = 10000, memory=10000,
                 LR_START = .003, LR_END = .0002, LR_DECAY = 200000,
                 EPS_START = 1, EPS_END = .05, EPS_DECAY = 200000,
                 GAMMA = 0.999, BATCH_SIZE = 8):
        self.env = Catan_Env(reward_function)
        self.game = self.env.game
        self.num_episodes = num_episodes
        self.device = device
        self.NEURAL_NET = model
        self.agent = DQNAgent(model, device, memory,
                                LR_START, LR_END, LR_DECAY,
                                EPS_START, EPS_END, EPS_DECAY,
                                GAMMA, BATCH_SIZE)
        self.agent_policy_net = self.agent.policy_net
        self.agent_target_net = self.agent.target_net
        self.optimizer = self.agent.optimizer
        self.memory = self.agent.memory
        self.steps_done = self.agent.steps_done
        self.cur_boardstate = None
        self.cur_vectorstate = None
        self.log = Log()

    def new_game(self):
        self.env.new_game()
        self.game = self.env.game
        self.cur_boardstate = state_changer(self.env)[0]
        self.cur_vectorstate = state_changer(self.env)[1]

    def select_action_agent0(self, boardstate, vectorstate):
        sample = random.random()
        eps_threshold = self.agent.EPS_END + (self.agent.EPS_START - self.agent.EPS_END)*math.exp(-1. * self.steps_done / self.agent.EPS_DECAY)
        lr = self.agent.LR_END + (self.agent.LR_START - self.agent.LR_END) * math.exp(-1. * self.steps_done / self.agent.LR_DECAY)
        # Update the learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        if sample > eps_threshold:
            with torch.no_grad():
                if self.game.cur_player == 0:
                    self.env.phase.actionstarted += 1
                    action = self.agent_policy_net(boardstate, vectorstate).max(1).indices.view(1,1)
                    if action >= 4*11*21:
                        final_action = int(action - 4*11*21 + 5)
                        position_y = 0
                        position_x = 0
                    else:
                        final_action = int(action)//(11*21)+1
                        position_y = (int(action) - ((final_action-1)*11*21))//21
                        position_x = int(action) % 21 
                    action_selecter(self.env, final_action, position_x, position_y)
                    self.log.action_counts[action] += 1
                    if self.env.phase.actionstarted >= 5:
                        action_selecter(self.env,5,0,0)
                    #return action
                
        else:
            final_action,position_x,position_y = random_assignment(self.env)
            if final_action > 4:
                action = final_action + 4*11*21 - 5
            else:
                action = (final_action-1)*11*21 + position_y*21 + position_x 
            self.log.random_action_counts[action] += 1
            self.game.random_action_made = 1
        action_tensor = torch.tensor([[action]], device=self.device, dtype=torch.long)
        return action_tensor
        

    def select_action_agent1(self):
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
        
        state_action_values = self.agent_policy_net(*state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.agent.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.agent_target_net(non_final_next_board_states, non_final_next_vector_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
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

    def train(self,PRINT_ACTIONS = True, gameStatePrintLevel = 0):
        start_time = time.time()
        for i_episode in range(self.num_episodes):
            self.new_game()
            time_new_start = time.time()
            print(i_episode)
            if i_episode % 50 == 49:
                self.agent_target_net.load_state_dict(self.agent_policy_net.state_dict())
            if i_episode % 20 == 19:
                torch.save(self.agent_policy_net.state_dict(), f'agent{i_episode}_policy_net_0_1_1.pth')
            for t in count():
                if self.game.cur_player == 1:
                    action = self.select_action_agent1()
                    if PRINT_ACTIONS:
                        InterpretActions(1,action, self.env, gameStatePrintLevel)
                    self.env.phase.actionstarted = 0
                    if self.env.phase.statechange == 1:
                        if self.game.is_finished == 1:
                            self.game.cur_player = 0
                            self.cur_boardstate =  state_changer(self.env)[0]
                            self.cur_vectorstate = state_changer(self.env)[1]
                            cur_boardstate = self.cur_boardstate.clone().detach().unsqueeze(0).to(self.device).float()        
                            cur_vectorstate = self.cur_vectorstate.clone().detach().unsqueeze(0).to(self.device).float()
                            next_board_state, next_vector_state, reward, done = state_changer(self.env)[0], state_changer(self.env)[1], self.env.phase.reward, self.game.is_finished
                            reward = torch.tensor([reward], device=self.device)
                            next_board_state = next_board_state.clone().detach().unsqueeze(0).to(self.device).float()
                            next_vector_state = next_vector_state.clone().detach().unsqueeze(0).to(self.device).float()
                            if done == 1:
                                self.env.phase.gamemoves = t
                                print("done0")
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
                            print("done1")
                            self.game.is_finished = 0
                            self.log.episode_durations.append(t+1)
                            break
                elif self.game.cur_player == 0:
                    self.cur_boardstate =  state_changer(self.env)[0]
                    self.cur_vectorstate = state_changer(self.env)[1]
                    cur_boardstate = self.cur_boardstate.clone().detach().unsqueeze(0).to(self.device).float()
                    cur_vectorstate = self.cur_vectorstate.clone().detach().unsqueeze(0).to(self.device).float()
                    action = self.select_action_agent0(cur_boardstate, cur_vectorstate)
                    if PRINT_ACTIONS:
                        InterpretActions(0,action, self.env, gameStatePrintLevel)
                    if self.env.phase.statechange == 1:
                        next_board_state, next_vector_state, reward, done = state_changer(self.env)[0], state_changer(self.env)[1], self.env.phase.reward, self.game.is_finished
                        reward = torch.tensor([reward], device=self.device)
                        next_board_state = next_board_state.clone().detach().unsqueeze(0).to(self.device).float()
                        next_vector_state = next_vector_state.clone().detach().unsqueeze(0).to(self.device).float()
                        if done == 1:
                            self.env.phase.gamemoves = t
                            print("done0")
                            next_board_state = None
                            next_vector_state = None
                        self.memory.push(cur_boardstate, cur_vectorstate,action,next_board_state, next_vector_state,reward)
                        self.cur_boardstate = next_board_state
                        self.cur_vectorstate = next_vector_state
                        self.optimize_model()

                        if done == 1:
                            self.env.phase.gamemoves = t
                            self.game.is_finished = 0
                            self.log.episode_durations.append(t+1)
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

            self.game.average_time.insert(0, time.time() - time_new_start)
            if len(self.game.average_time) > 10:
                self.game.average_time.pop(10)
            self.game.average_moves.insert(0, t+1)
            if len(self.game.average_moves) > 10:
                self.game.average_moves.pop(10)
            if i_episode > 1:
                self.game.average_legal_moves_ratio.insert(0, (self.env.phase.statechangecount - statechangecountprevious)/t)
                if len(self.game.average_legal_moves_ratio) > 20:
                    self.game.average_legal_moves_ratio.pop(20)
            statechangecountprevious = self.env.phase.statechangecount
            self.env.phase.statechange = 0
            self.game.random_action_made = 0
            self.env.phase.reward = 0
        
        elapsed_time = time.time() - start_time
        print('Complete')
        print(f'steps over {self.num_episodes} episodes: {self.steps_done}')
        print(f'Elapsed time: {elapsed_time}')
        print(f'Optimizer steps: {len(self.game.average_q_value_loss)}')
        print(f'Optimizer Loss avg: {np.mean(self.game.average_q_value_loss)}')

def main(MEMORY,MODEL_SELECT,REWARD_FUNCTION,NUM_EPISODES,
         LR_START,LR_END,LR_DECAY,EPS_START,EPS_END,EPS_DECAY,
         GAMMA,BATCH_SIZE,PRINT_ACTIONS):
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODEL_SELECT =='Large':
        from DQN.Neural_Networks.DQN_Big import DQN as dqn
    elif MODEL_SELECT =='Medium':
        from DQN.Neural_Networks.DQN_Medium import DQN as dqn
    else: 
        from DQN.Neural_Networks.DQN_Small import DQN as dqn
    model = dqn()
    model.to(device)

    training = Catan_Training(REWARD_FUNCTION, device, model, num_episodes=NUM_EPISODES, memory=MEMORY,
                              LR_START=LR_START, LR_END=LR_END, LR_DECAY=LR_DECAY,
                              EPS_START=EPS_START, EPS_END=EPS_END,
                              EPS_DECAY=EPS_DECAY, GAMMA=GAMMA, BATCH_SIZE=BATCH_SIZE)
    training.train(PRINT_ACTIONS)

main(MEMORY,MODEL_SELECT,REWARD_FUNCTION,NUM_EPISODES,LR_START,LR_END,LR_DECAY,EPS_START,EPS_END,EPS_DECAY,GAMMA,BATCH_SIZE,PRINT_ACTIONS)