import numpy as np
import matplotlib.pyplot as plt

# Grid dimensions based on the initial setup
_NUM_ROWS = 11
_NUM_COLS = 21

# Initialize the grid arrays for settlement availability and tile locations
settlements_available = np.zeros((_NUM_ROWS, _NUM_COLS))
TILES_POSSIBLE = np.zeros((_NUM_ROWS, _NUM_COLS))

# Settlement building function based on provided logic
def settlements_building():
    for i in range(0, _NUM_ROWS, 2):
        for j in range(-1 + abs(5 - i), 23 - abs(5 - i), 2):
            if 0 <= j < _NUM_COLS:  # Ensure indices are within grid bounds
                settlements_available[i][j] = 1

# Tile building function based on provided logic
def tiles_building():
    for i in range(1, _NUM_ROWS - 1, 2):
        for j in range(2 + abs(5 - i), _NUM_COLS - abs(5 - i), 4):
            TILES_POSSIBLE[i][j] = 1

# Run the functions to set tile and settlement positions
settlements_building()
tiles_building()

# Plot setup with tiles and settlements
# Enhanced visualization with larger font sizes, squares, and dots

# Plot setup with increased marker sizes and font size
plt.figure(figsize=(12, 6))
plt.imshow(np.zeros((_NUM_ROWS, _NUM_COLS)), cmap="Blues", aspect="auto", alpha=0.1)  # Light background grid

# Plot tile locations in larger light blue squares
tile_positions = np.where(TILES_POSSIBLE == 1)
plt.scatter(tile_positions[1], tile_positions[0], color="lightblue", marker="s", s=250, label="Tile")

# Plot settlement locations with larger red dots
settlement_positions = np.where(settlements_available == 1)
plt.scatter(settlement_positions[1], settlement_positions[0], color="red", marker="o", s=100, label="Available Settlement")

# Enhanced plot details
plt.title("Grid Representation: Available Settlement Locations and Tiles", fontsize=18)
plt.xlabel("Columns", fontsize=14)
plt.ylabel("Rows", fontsize=14)
plt.legend(loc="upper right", fontsize=12)
plt.show()

