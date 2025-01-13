import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_nd

def IsPathOpen(map, a, b):
    """
    Checks whether the path from point a to point b on the map is obstacle-free.

    Parameters:
        map (np.ndarray): 2D array representing the map (0: free, 1: obstacle).
        a (tuple): Starting point (row, col).
        b (tuple): Ending point (row, col).

    Returns:
        bool: True if the path is obstacle-free, False otherwise.
    """
    # Generate the line between points a and b
    line_indices = line_nd(a, b, integer=True)

    # Plot the map and visualize the search process
    plt.figure(figsize=(8, 6))
    plt.imshow(map,  origin='upper')
    plt.title("Path Search Visualization")
    
    # Check all points on the line
    is_obstacle_free = True
    for x, y in zip(*line_indices):
        if map[x, y] == 255:  # Open space
            plt.plot(y, x, 'w.')  # Mark open space as white
        else:  # Obstacle
            plt.plot(y, x, 'r.')  # Mark obstacle as red
            is_obstacle_free = False
    
    plt.show()
    return is_obstacle_free

#Test IsPathOpen
from skimage.draw import line_nd, random_shapes
map, labels = random_shapes((200,300),20,5,num_channels=1)

# Define start and end points
a = (10, 10)
b = (190, 290)

# Check if the path is open
result = IsPathOpen(map, a, b)
print("Path open:", result)

