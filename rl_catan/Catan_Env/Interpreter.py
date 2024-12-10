import math 
def InterpretActions(player, selected_action, env, action_was_random = True, log_file = None, gameStatePrintLevel = 0):
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
        6: "trades lumber for wool",
        7: "trades lumber for grain",
        8: "trades lumber for brick",
        9: "trades lumber for ore",
        10: "trades wool for lumber",
        11: "trades wool for grain",
        12: "trades wool for brick",
        13: "trades wool for ore",
        14: "trades grain for lumber",
        15: "trades grain for wool",
        16: "trades grain for brick",
        17: "trades grain for ore",
        18: "trades brick for lumber",
        19: "trades brick for wool",
        20: "trades brick for grain",
        21: "trades brick for ore",
        22: "trades ore for lumber",
        23: "trades ore for wool",
        24: "trades ore for grain",
        25: "trades ore for brick",
        26: "buys a dev card",
        27: "activates a knight dev card",
        28: "activates a road builder dev card",
        29: "activates a year of plenty dev card",
        30: "activates a monopoly dev card",
        31: "uses year of plenty for wood",
        32: "uses year of plenty for wool",
        33: "uses year of plenty for grain",
        34: "uses year of plenty for brick",
        35: "uses year of plenty for ore",
        36: "uses monopoly for lumber",
        37: "uses monopoly for wool",
        38: "uses monopoly for grain",
        39: "uses monopoly for brick",
        40: "uses monopoly for ore",
        }
    
    # Fetch and print the appropriate message
    message = action_messages.get(final_action, "Unknown action")
    
    if log_file is not None:
        if action_was_random:
            message = "Randomly selected " + message
        else:
            message = "Policy selected " + message
        log_file.write(f"Player: {player}, {message}\nReward for the agent: {env.phase.reward}\n")
        if final_action == 5:
            if gameStatePrintLevel == 1:
                log_file.write(f"\nPlayer 0 Stats:")
                log_file.write(env.player0)
                log_file.write(f"\nPlayer 1 Stats:")
                log_file.write(env.player1)
            elif gameStatePrintLevel == 2:
                log_file.write(f"\nBoard state:")
                log_file.write(env.board)
            elif gameStatePrintLevel == 3:
                log_file.write(f"\nPlayer 0 Stats:")
                log_file.write(env.player0)
                log_file.write(f"\nPlayer 1 Stats:")
                log_file.write(env.player1)
                log_file.write(f"\nBoard state:")
                log_file.write(env.board)


