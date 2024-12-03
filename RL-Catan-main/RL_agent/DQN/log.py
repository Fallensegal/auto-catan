import math
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(grandparent_dir)
from Configurations import *
from Catan_Env.catan_env import Catan_Env
from .config import *


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

        self.action_counts = []
        self.random_action_counts = []
        self.episode_durations = []




log = Log()
log.action_counts = [0] * TOTAL_ACTIONS
log.random_action_counts = [0] * TOTAL_ACTIONS


def logging(num_episode):
    player0 = Catan_Env().players[0]
    player1 = Catan_Env().players[1]

    phase = Catan_Env().phase
    random_testing = Catan_Env().random_testing

    game = Catan_Env().game

    player0_log = Catan_Env().player0_log
    player1_log = Catan_Env().player1_log

    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * log.steps_done / EPS_DECAY)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(log.action_counts))), y=log.action_counts, mode='markers', name='Action Counts'))
    fig.add_trace(go.Scatter(x=list(range(len(log.random_action_counts))), y=log.random_action_counts, mode='markers', name='Random Action Counts'))
    phase.victoryreward = 0
    phase.victoryreward = 0
    phase.illegalmovesreward = 0
    phase.legalmovesreward = 0