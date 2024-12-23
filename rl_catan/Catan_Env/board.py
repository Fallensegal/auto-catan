import numpy as np

_NUM_ROWS = 11
_NUM_COLS = 21

class Board:
    def __init__(self):
        #_______________________input_________________________
        self.tiles_lumber = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_wool = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_grain = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_brick = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_ore = np.zeros((_NUM_ROWS, _NUM_COLS))

        self.tiles_probability_1 = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_probability_2 = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_probability_3 = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_probability_4 = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_probability_5 = np.zeros((_NUM_ROWS, _NUM_COLS))

        self.rober_position = np.zeros((_NUM_ROWS, _NUM_COLS))

        self.harbor_lumber = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_wool = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_grain = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_brick = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_ore = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_three_one = np.zeros((_NUM_ROWS, _NUM_COLS))

        #_________________ game specific ________________
        #board
        self.ZEROBOARD = np.zeros((_NUM_ROWS, _NUM_COLS))
        #tiles
        self.TILES_POSSIBLE = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_dice = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_dice_probabilities = np.zeros((_NUM_ROWS, _NUM_COLS))

        #settlements
        self.settlements_free = np.ones((_NUM_ROWS, _NUM_COLS))
        self.settlements_available = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.settlements_used = np.zeros((_NUM_ROWS, _NUM_COLS))

        #roads
        self.roads_available = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.roads_free = np.zeros((_NUM_ROWS, _NUM_COLS))

        #harbors
        self.harbors_possible = np.zeros((9, 2, 2))

        #longest road
        self.longest_road = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.increasing_roads = np.zeros((_NUM_ROWS, _NUM_COLS))

    def find_non_zero_elements(self, matrix):
        non_zero_elements = np.argwhere(matrix != 0)
        result = []
        for element in non_zero_elements:
            row, col = element
            result.append(f"Row: {row}, Column: {col}")
        return '\n'.join(result)
    
    def find_zero_elements(self, matrix):
        zero_elements = np.argwhere(matrix == 0)
        result = []
        for element in zero_elements:
            row, col = element
            result.append(f"Row: {row}, Column: {col}")
        return '\n'.join(result)

    def __str__(self):
        return (#f"Tiles:\n"
                #f"Lumber: {self.find_non_zero_elements(self.tiles_lumber)}\n"
                #f"Wool: {self.find_non_zero_elements(self.tiles_wool)}\n"
                #f"Grain: {self.find_non_zero_elements(self.tiles_grain)}\n"
                #f"Brick: {self.find_non_zero_elements(self.tiles_brick)}\n"
                #f"Ore: {self.find_non_zero_elements(self.tiles_ore)}\n"
                #f"Tile Probabilities:\n"
                #f"Probability 1: {self.find_non_zero_elements(self.tiles_probability_1)}\n"
                #f"Probability 2: {self.find_non_zero_elements(self.tiles_probability_2)}\n"
                #f"Probability 3: {self.find_non_zero_elements(self.tiles_probability_3)}\n"
                #f"Probability 4: {self.find_non_zero_elements(self.tiles_probability_4)}\n"
                #f"Probability 5: {self.find_non_zero_elements(self.tiles_probability_5)}\n"
                f"Robber Position:\n"
                f"{self.find_non_zero_elements(self.rober_position)}\n"
                #f"Harbors:\n"
                #f"Lumber: {self.find_non_zero_elements(self.harbor_lumber)}\n"
                #f"Wool: {self.find_non_zero_elements(self.harbor_wool)}\n"
                #f"Grain: {self.find_non_zero_elements(self.harbor_grain)}\n"
                #f"Brick: {self.find_non_zero_elements(self.harbor_brick)}\n"
                #f"Ore: {self.find_non_zero_elements(self.harbor_ore)}\n"
                #f"3:1: {self.find_non_zero_elements(self.harbor_three_one)}\n"
                f"Settlements:\n"
                #f"Free:\n{self.find_non_zero_elements(self.settlements_free)}\n"
                #f"Available:\n{self.find_non_zero_elements(self.settlements_available)}\n"
                f"Used:\n{self.find_zero_elements(self.settlements_used)}\n"
                #f"Roads:\n"
                #f"Available:\n{self.find_non_zero_elements(self.roads_available)}\n"
                #f"Free:\n{self.find_zero_elements(self.roads_free)}\n"
                f"Longest Road:\n"
                f"Current:\n{self.find_non_zero_elements(self.longest_road)}\n")
                #f"Increasing:\n{self.find_non_zero_elements(self.increasing_roads)}")

    def harbors_building(self):
        # Define harbor locations

        self.harbors_possible[0] = [[0,4],[0,6]]
        self.harbors_possible[1] = [[0,10],[0,12]]
        self.harbors_possible[2] = [[2,16],[2,18]]
        self.harbors_possible[3] = [[2,2],[4,2]]
        self.harbors_possible[4] = [[6,2],[8,2]]
        self.harbors_possible[5] = [[10,4],[10,6]]
        self.harbors_possible[6] = [[10,10],[10,12]]
        self.harbors_possible[7] = [[8,16],[8,18]]
        self.harbors_possible[8] = [[4,20],[6,20]]

    
    def tiles_buidling(self):
        for i in range(1,10,2):
            for j in range(2 + abs(5-i),20 - abs(5-i),4):
                self.TILES_POSSIBLE[i][j] = 1


    def settlements_building(self):
        for i in range(0,11,2):
            for j in range(-1 + abs(5-i),23 - abs(5-i),2):
                self.settlements_available[i][j] = 1  
                


    def roads_building(self):
        for i in range(0,10,1):
            for j in range(0,20,1):
                if self.settlements_available[i + 1][j] == 1 and self.settlements_available[i - 1][j] == 1:
                    self.roads_available[i][j] = 1
                elif self.settlements_available[i + 1][j + 1] == 1 and self.settlements_available[i - 1][j + 1] == 1:
                    self.roads_available[i][j+1] = 1
                elif self.settlements_available[i][j + 1] == 1 and self.settlements_available[i][j - 1] == 1:
                    self.roads_available[i][j] = 1
                elif self.settlements_available[i + 1][j + 1] == 1 and self.settlements_available[i + 1][j - 1] == 1:
                    self.roads_available[i+1][j] = 1


        self.roads_available = self.roads_available*(1-self.TILES_POSSIBLE)