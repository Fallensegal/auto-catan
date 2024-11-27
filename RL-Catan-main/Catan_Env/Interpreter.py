import math 
def InterpretActions(player, selected_action):
    selected_action = selected_action.item()
    if selected_action >= 4*11*21:
        final_action = selected_action - 4*11*21 + 5
        position_y = 0
        position_x = 0
    else:
        final_action = selected_action//(11*21)+1
        position_y = (selected_action - ((final_action-1)*11*21))//21
        position_x = selected_action % 21 

    # Action message mapping
    action_messages = {
        1: f"move Robber to position: {position_y}, {position_x}",
        2: f"build road at position: {position_y}, {position_x}",
        3: f"build settlement at position: {position_y}, {position_x}",
        4: f"build city at position: {position_y}, {position_x}",
        5: "ends turn",
        6: "keeps lumber",
        7: "keeps wool",
        8: "keeps grain",
        9: "keeps brick",
        10: "keeps ore",
        11: "trades lumber for wool",
        12: "trades lumber for grain",
        13: "trades lumber for brick",
        14: "trades lumber for ore",
        15: "trades wool for lumber",
        16: "trades wool for grain",
        17: "trades wool for brick",
        18: "trades wool for ore",
        19: "trades grain for lumber",
        20: "trades grain for wool",
        21: "trades grain for brick",
        22: "trades grain for ore",
        23: "trades brick for lumber",
        24: "trades brick for wool",
        25: "trades brick for grain",
        26: "trades brick for ore",
        27: "trades ore for lumber",
        28: "trades ore for wool",
        29: "trades ore for grain",
        30: "trades ore for brick",
        31: "buys a dev card",
        32: "activates a knight dev card",
        33: "activates a road builder dev card",
        34: "activates a year of plenty dev card",
        35: "activates a monopoly dev card",
        36: "uses year of plenty for wood",
        37: "uses year of plenty for wool",
        38: "uses year of plenty for grain",
        39: "uses year of plenty for brick",
        40: "uses year of plenty for ore",
        41: "uses monopoly for lumber",
        42: "uses monopoly for wool",
        43: "uses monopoly for grain",
        44: "uses monopoly for brick",
        45: "uses monopoly for ore",
    }
    
    # Fetch and print the appropriate message
    message = action_messages.get(final_action, "Unknown action")
    print(f"Player: {player}, {message}")
