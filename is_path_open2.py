import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_nd, random_shapes

def IsPathOpen(map, a, b):
    """
    Checks whether the path from point a to point b on the map is obstacle-free.

    Parameters:
        map (np.ndarray): 2D array representing the map (0: obstacle, 255: free space).
        a (tuple): Starting point (row, col).
        b (tuple): Ending point (row, col).

    Returns:
        bool: True if the path is obstacle-free, False otherwise.
    """
    # Generate the line between points a and b
    line_indices = line_nd(a, b, integer=True)

    # Check all points on the line
    for x, y in zip(*line_indices):
        if map[x, y] < 255:  # Obstacle detected (non-background area)
            return False
    return True

def visualize_map_with_path(map, path, color='w.'):
    """
    Visualize the map and the given path.

    Parameters:
        map (np.ndarray): 2D array representing the map.
        path (list): List of points (row, col) representing the path.
        color (str): Color and marker style for the path.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(map, cmap='gray', origin='upper')
    plt.title("Map with Path")
    for point in path:
        plt.plot(point[1], point[0], color)
    plt.show()

# Generate a random map with obstacles
map, _ = random_shapes((200, 300), max_shapes=20, min_shapes=5, num_channels=1)
map = (map == 255).astype(np.uint8) * 255  # Convert to binary: 255 (free), 0 (obstacle)

# Define nodes for testing
nodes = [(10, 10), (50, 50), (100, 150), (190, 290), (150, 30)]
edges = []

# Check and add edges if obstacle-free
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        if IsPathOpen(map, nodes[i], nodes[j]):
            edges.append((nodes[i], nodes[j]))

# Visualize the map and edges
plt.figure(figsize=(8, 6))
plt.imshow(map, cmap='gray', origin='upper')
plt.title("Map with Edges")
for edge in edges:
    start, end = edge
    line_indices = line_nd(start, end, integer=True)
    plt.plot(line_indices[1], line_indices[0], 'w-')  # Draw the line in white
plt.scatter([node[1] for node in nodes], [node[0] for node in nodes], c='red', label="Nodes")
plt.legend()
plt.show()

