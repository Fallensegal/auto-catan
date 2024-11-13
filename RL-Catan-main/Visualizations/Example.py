import numpy as np
import matplotlib.pyplot as plt

# Constants based on the user's code
_NUM_ROWS = 11
_NUM_COLS = 21

# Board class as per user's implementation
class Board: 
    def __init__(self):
        # Tile availability setup
        self.TILES_POSSIBLE = np.zeros((_NUM_ROWS, _NUM_COLS))
    
    def tiles_buidling(self):
        # Defining valid tile positions in a hexagonal pattern
        for i in range(1, 10, 2):
            for j in range(2 + abs(5 - i), 20 - abs(5 - i), 4):
                self.TILES_POSSIBLE[i][j] = 1

# Create board instance and set up tiles
board = Board()
board.tiles_buidling()

# Visualization
plt.figure(figsize=(10, 5))
plt.imshow(board.TILES_POSSIBLE, cmap="Blues", aspect="auto")
plt.title("Catan Board Tile Layout (Valid Tiles in Blue)")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.colorbar(label="Tile Presence (1 = Tile, 0 = No Tile)")
plt.show()