import numpy as np

_NUM_ROWS = 11
_NUM_COLS = 21

class Player:
    def __init__(self):
        #________________________input board__________________________
        self.settlements = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.roads = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.cities = np.zeros((_NUM_ROWS, _NUM_COLS))       
        #_______________________input vector__________________________
        self.resource_lumber = 0
        self.resource_wool = 0
        self.resource_grain = 0
        self.resource_brick = 0
        self.resource_ore = 0

        self.settlements_left = 5
        self.roads_left = 15
        self.cities_left = 4

        self.army_size = 0

        self.knight_cards_old = 0
        self.victorypoints_cards_old = 0
        self.yearofplenty_cards_old = 0
        self.monopoly_cards_old = 0
        self.roadbuilding_cards_old = 0

        self.knight_cards_new = 0
        self.victorypoints_cards_new = 0
        self.yearofplenty_cards_new = 0
        self.monopoly_cards_new = 0
        self.roadbuilding_cards_new = 0

        self.harbor_lumber = 0
        self.harbor_wool = 0
        self.harbor_grain = 0
        self.harbor_brick = 0
        self.harbor_ore = 0
        self.harbor_three_one = 0

        self.largest_army = 0

        self.roads_connected = 0
        self.longest_road = 0

        self.knight_cards_played = 0

        self.victorypoints_before = 0
        self.victorypoints = 0

        self.development_card_played = 0
        self.knight_move_pending = 0
        self.monopoly_move_pending = 0
        self.roadbuilding_move_pending = 0
        self.roadbuilding1 = 0
        self.yearofplenty_move_pending = 0

        self.yearofplenty_started = 0
        self.yearofplenty1 = 0
        self.yearofplenty2 = 0

        self.discard_resources_turn = 0
        self.num_discarded_resources = 0
        self.discard_first_time = 0
        self.total_resources = 0


        #__________________game-specific resource_____________
        #roads
        self.roads_possible = np.zeros((_NUM_ROWS, _NUM_COLS))

        #rewards 
        self.rewards_possible = np.zeros((_NUM_ROWS,_NUM_COLS))

        self.roadbuilding_d = 0
        self.roadbuilding_e = 0

        self.wins = 0

    def find_non_zero_elements(self, matrix):
        non_zero_elements = np.argwhere(matrix != 0)
        result = []
        for element in non_zero_elements:
            row, col = element
            result.append(f"Row: {row}, Column: {col}")
        return '\n'.join(result)

    def __str__(self):
        return (f"Resources:\n"
                f"Lumber: {self.resource_lumber}, Wool: {self.resource_wool}, Grain: {self.resource_grain}, "
                f"Brick: {self.resource_brick}, Ore: {self.resource_ore}\n"
                f"Development Cards (New):\n"
                f"Knights: {self.knight_cards_new}, Victory Points: {self.victorypoints_cards_new}, "
                f"Year of Plenty: {self.yearofplenty_cards_new}, Monopoly: {self.monopoly_cards_new}, "
                f"Road Building: {self.roadbuilding_cards_new}\n"
                f"Development Cards (Old):\n"
                f"Knights: {self.knight_cards_old}, Victory Points: {self.victorypoints_cards_old}, "
                f"Year of Plenty: {self.yearofplenty_cards_old}, Monopoly: {self.monopoly_cards_old}, "
                f"Road Building: {self.roadbuilding_cards_old}\n"
                f"Victory Points: {self.victorypoints}, Largest Army: {self.largest_army}, Longest Road: {self.longest_road}\n"
                f"Settlements Left: {self.settlements_left}, Cities Left: {self.cities_left}, Roads Left: {self.roads_left}\n"
                #f"Total Resources: {self.total_resources}\n"
                f"Board State:\n"
                f"Settlements:\n{self.find_non_zero_elements(self.settlements)}\n"
                f"Roads:\n{self.find_non_zero_elements(self.roads)}\n"
                f"Cities:\n{self.find_non_zero_elements(self.cities)}")

    class Action: 
        def __init__(self):
            #________________________Output board_______________
            self.rober_move = np.zeros((_NUM_ROWS,_NUM_COLS))

            self.road_place = np.zeros((_NUM_ROWS,_NUM_COLS))
            self.settlement_place = np.zeros((_NUM_ROWS,_NUM_COLS))
            self.city_place = np.zeros((_NUM_ROWS,_NUM_COLS))

            #______________________Vector_____________________
            self.end_turn = 0

            #how many development cards the agent wants to buy 
            self.development_card_buy = 0

            #Play a development card
            self.knight_cards_activate = 0 
            self.road_building_cards_activate = 0
            self.monopoly_cards_activate = 0
            self.yearofplenty_cards_activate = 0


            #Which resources do you want to take (Chooses twice)
            self.yearofplenty_lumber = 0
            self.yearofplenty_wool = 0
            self.yearofplenty_grain = 0
            self.yearofplenty_brick = 0
            self.yearofplenty_ore = 0

            #Which resource do you want to take when playing monopoly
            self.monopoly_lumber = 0
            self.monopoly_wool = 0
            self.monopoly_grain = 0
            self.monopoly_brick = 0
            self.monopoly_ore = 0
    class Trading: 
        def __init__(self):
            self.give_lumber_get_wool = 0
            self.give_lumber_get_grain = 0
            self.give_lumber_get_brick = 0
            self.give_lumber_get_ore  = 0
            self.give_wool_get_lumber = 0
            self.give_wool_get_grain = 0
            self.give_wool_get_brick = 0
            self.give_wool_get_ore = 0
            self.give_grain_get_lumber = 0
            self.give_grain_get_wool = 0
            self.give_grain_get_brick = 0
            self.give_grain_get_ore = 0
            self.give_brick_get_lumber = 0
            self.give_brick_get_wool = 0
            self.give_brick_get_grain = 0
            self.give_brick_get_ore = 0
            self.give_ore_get_lumber = 0
            self.give_ore_get_wool = 0
            self.give_ore_get_grain = 0
            self.give_ore_get_brick = 0
    class Log:
        def __init__(self):
            #The average is taken over the last 10 games 
            self.average_victory_points = []
            self.average_resources_found = []
            #self.average_resources_found_move = 0 | I can calculate this  
            self.final_board_state = 0 #tommorow
            self.AI_function_calls = 0 #same here
            self.successful_AI_function_calls = 0 #same here
            self.average_development_cards_bought = []
            self.average_roads_built = []
            self.average_settlements_built = []
            self.average_cities_built = []
            self.average_knights_played = []
            self.average_development_cards_used = [] #victory point cards are seen as automatically used
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

            self.old_total_resources_found = 0
            self.old_total_development_cards_bought = 0
            self.old_total_roads_built = 0
            self.old_total_settlements_built = 0
            self.old_total_cities_built = 0
            self.old_total_development_cards_used = 0
            self.old_total_resources_traded = 0
            self.old_total_knights_played = 0
            
        def UpdateOldLog(self):
            self.old_total_self_resources_found          = self.total_resources_found
            self.old_total_self_development_cards_bought = self.total_development_cards_bought
            self.old_total_self_roads_built              = self.total_roads_built             
            self.old_total_self_settlements_built        = self.total_settlements_built
            self.old_total_self_cities_built             = self.total_cities_built
            self.old_total_self_development_cards_used   = self.total_development_cards_used
            self.old_total_self_resources_traded         = self.total_resources_traded
            self.old_total_self_knights_played           = self.total_knights_played
    