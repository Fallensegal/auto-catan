import numpy as np

def action_selecter(env,selected_action, selected_position_x = 0, selected_position_y = 0):
    board = env.board
    action = env.player_action[env.game.cur_player]
    trading = env.player_trading[env.game.cur_player]

    env.total_step += 1

    action.rober_move = action.rober_move * board.ZEROBOARD
    action.road_place = action.road_place * board.ZEROBOARD
    action.settlement_place = action.settlement_place * board.ZEROBOARD
    action.city_place = action.city_place * board.ZEROBOARD
    action.end_turn = 0

    trading.give_lumber_get_wool = 0
    trading.give_lumber_get_grain = 0
    trading.give_lumber_get_brick = 0
    trading.give_lumber_get_ore  = 0
    trading.give_wool_get_lumber = 0
    trading.give_wool_get_grain = 0
    trading.give_wool_get_brick = 0
    trading.give_wool_get_ore = 0
    trading.give_grain_get_lumber = 0
    trading.give_grain_get_wool = 0
    trading.give_grain_get_brick = 0
    trading.give_grain_get_ore = 0
    trading.give_brick_get_lumber = 0
    trading.give_brick_get_wool = 0
    trading.give_brick_get_grain = 0
    trading.give_brick_get_ore = 0
    trading.give_ore_get_lumber = 0
    trading.give_ore_get_wool = 0
    trading.give_ore_get_grain = 0
    trading.give_ore_get_brick = 0
    
    action.development_card_buy = 0
    action.knight_cards_activate = 0
    action.road_building_cards_activate = 0
    action.yearofplenty_cards_activate = 0
    action.monopoly_cards_activate = 0
    action.yearofplenty_lumber = 0
    action.yearofplenty_wool = 0
    action.yearofplenty_grain = 0
    action.yearofplenty_brick = 0
    action.yearofplenty_ore = 0
    action.monopoly_lumber = 0
    action.monopoly_wool = 0
    action.monopoly_grain = 0
    action.monopoly_brick = 0
    action.monopoly_ore = 0

    if selected_action == 1:  
        action.rober_move[selected_position_y] [selected_position_x] = 1
    if selected_action == 2:
        action.road_place[selected_position_y] [selected_position_x] = 1
    if selected_action == 3:
        action.settlement_place[selected_position_y] [selected_position_x] = 1

    if selected_action == 4:
        action.city_place[selected_position_y] [selected_position_x] = 1
    if selected_action == 5:
        action.end_turn = 1  
    if selected_action == 6:  
        trading.give_lumber_get_wool = 1
    if selected_action == 7:
        trading.give_lumber_get_grain = 1
    if selected_action == 8:
        trading.give_lumber_get_brick = 1
    if selected_action == 9:
        trading.give_lumber_get_ore = 1
    if selected_action == 10:
        trading.give_wool_get_lumber = 1
    if selected_action == 11:
        trading.give_wool_get_grain = 1
    if selected_action == 12:
        trading.give_wool_get_brick = 1
    if selected_action == 13:
        trading.give_wool_get_ore = 1
    if selected_action == 14:
        trading.give_grain_get_lumber = 1
    if selected_action == 15:
        trading.give_grain_get_wool = 1
    if selected_action == 16:
        trading.give_grain_get_brick = 1
    if selected_action == 17:
        trading.give_grain_get_ore = 1
    if selected_action == 18:
        trading.give_brick_get_lumber = 1
    if selected_action == 19:
        trading.give_brick_get_wool = 1
    if selected_action == 20:
        trading.give_brick_get_grain = 1
    if selected_action == 21:
        trading.give_brick_get_ore = 1
    if selected_action == 22:
        trading.give_ore_get_lumber = 1
    if selected_action == 23:
        trading.give_ore_get_wool = 1
    if selected_action == 24:
        trading.give_ore_get_grain = 1
    if selected_action == 25:
        trading.give_ore_get_brick = 1
    if selected_action == 26:
        action.development_card_buy = 1
    if selected_action == 27:
        action.knight_cards_activate = 1
    if selected_action == 28:
        action.road_building_cards_activate = 1
    if selected_action == 29:
        action.yearofplenty_cards_activate = 1
    if selected_action == 30:
        action.monopoly_cards_activate = 1
    if selected_action == 31:
        action.yearofplenty_lumber = 1
    if selected_action == 32:
        action.yearofplenty_wool = 1
    if selected_action == 33:
        action.yearofplenty_grain = 1
    if selected_action == 34:
        action.yearofplenty_brick = 1
    if selected_action == 35:
        action.yearofplenty_ore = 1
    if selected_action == 36:
        action.monopoly_lumber = 1
    if selected_action == 37:
        action.monopoly_wool = 1
    if selected_action == 38:
        action.monopoly_grain = 1
    if selected_action == 39:
        action.monopoly_brick = 1
    if selected_action == 40:
        action.monopoly_ore = 1      

    env.action_executor()

class Random: 
    def __init__(self):
        self.random_action = 0
        self.random_position_x = 0
        self.random_position_y = 0

random_agent = Random()

def random_assignment(env):
    legal_actions = env.checklegalmoves()

    legal_indices = np.where(legal_actions == 1)[1]

    randomaction = np.random.choice(legal_indices)
    if randomaction >= 4*11*21:
            final_action = randomaction - 4*11*21 + 5
            position_y = 0
            position_x = 0
    else:
        final_action = randomaction//(11*21)+1
        position_y = (randomaction - ((final_action-1)*11*21))//21
        position_x = randomaction % 21 

    action_selecter(env, final_action, position_x, position_y)



    #print(randomaction)

    #random_agent.random_action = np.random.choice(np.arange(1,46), p=[1/14, 2/14, 2/14, 2/14, 1/14, 1/35, 1/35, 1/35, 1/35, 1/35, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/14, 1/28, 1/28, 1/28, 1/28, 1/140, 1/140, 1/140, 1/140, 1/140, 1/700, 1/700, 1/700, 1/700, 1/700])    
    #random_agent.random_position_y = np.random.choice(np.arange(0,11))
    #random_agent.random_position_x = np.random.choice(np.arange(0,21))
    #action_selecter(env,random_agent.random_action, random_agent.random_position_x, random_agent.random_position_y)
    return(final_action, position_x, position_y)