import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_nd, random_shapes


# Function to check if the path between two points is obstacle-free
def IsPathOpen(map, a, b):
    """
    Check if the line from point a to b on the map is obstacle-free.

    Args:
    - map: 2D array representing the environment (0 for free space, >0 for obstacles).
    - a: Tuple (x, y) representing the starting point.
    - b: Tuple (x, y) representing the ending point.

    Returns:
    - True if the path is obstacle-free, False otherwise.
    """
    # Generate the discrete points along the line
    x, y = line_nd(a, b, endpoint=True)

    # Check each point on the line
    for i in range(len(x)):
        # Ensure points are within map bounds
        if not (0 <= x[i] < map.shape[0] and 0 <= y[i] < map.shape[1]):
            return False  # Out of bounds
        
        # Check for obstacles
        if map[x[i], y[i]] < 255:  # Obstacle detected (values < 255 are obstacles)
            return False

    return True


# Function to find the closest node in the tree
def find_closest_node(graph, qrand):
    min_dist = float("inf")
    closest_node = None
    for node in graph.keys():
        dist = np.sqrt((node[0] - qrand[0])**2 + (node[1] - qrand[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node


# RRT with controlled growth, collision checking, and integer pixel snapping
def grow_rrt_with_pixel_snapping(map, start, goal, Dq, iterations=2500):
    """
    Grow an RRT tree with controlled growth, straight-line collision checking,
    and snapped nodes to integer pixel coordinates.

    Args:
    - map: 2D binary array (0 for obstacles, 255 for free space).
    - start: Tuple (x, y) for the start node.
    - goal: Tuple (x, y) for the goal node.
    - Dq: Step size for controlled growth.
    - iterations: Maximum number of iterations.

    Returns:
    - G: Graph structure of the RRT.
    - parent: Parent dictionary for path reconstruction.
    """
    G = {start: []}  # Initialize the graph with the start node
    parent = {start: None}  # Track parent nodes for path reconstruction

    plt.imshow(map, cmap='gray')
    plt.plot(start[1], start[0], 'go', markersize=8)  # Start in green
    plt.plot(goal[1], goal[0], 'bo', markersize=8)   # Goal in blue
    plt.ion()  # Turn on interactive mode
    plt.show()

    for _ in range(iterations):
        # Sample a random configuration
        if np.random.random() < 0.1:  # 10% chance to bias toward the goal
            qrand = goal
        else:
            qrand = (np.random.randint(0, map.shape[0]), np.random.randint(0, map.shape[1]))

        # Find the closest node in the tree
        qnear = find_closest_node(G, qrand)

        # Compute the direction and step toward qrand
        theta = np.arctan2(qrand[0] - qnear[0], qrand[1] - qnear[1])
        qnew_float = (qnear[0] + Dq * np.sin(theta), qnear[1] + Dq * np.cos(theta))

        # Snap qnew to the nearest pixel
        qnew = (round(qnew_float[0]), round(qnew_float[1]))

        # Check if qnew is within bounds and if the straight line is obstacle-free
        if 0 <= qnew[0] < map.shape[0] and 0 <= qnew[1] < map.shape[1] and IsPathOpen(map, qnear, qnew):
            # Add the new node to the tree
            G[qnear].append((qnew, np.sqrt((qnew[0] - qnear[0])**2 + (qnew[1] - qnear[1])**2)))
            G[qnew] = []
            parent[qnew] = qnear

            # Visualize the edge in the tree
            plt.plot([qnear[1], qnew[1]], [qnear[0], qnew[0]], 'ro-', linewidth=2, markersize=3)
            plt.pause(0.01)  # Pause for animation

            # Check if qnew is close enough to the goal and the path is obstacle-free
            if np.sqrt((qnew[0] - goal[0])**2 + (qnew[1] - goal[1])**2) < Dq and IsPathOpen(map, qnew, goal):
                G[qnew].append((goal, np.sqrt((goal[0] - qnew[0])**2 + (goal[1] - qnew[1])**2)))
                G[goal] = []
                parent[goal] = qnew
                print("Goal reached!")
                break

    plt.ioff()  # Turn off interactive mode
    plt.show()

    return G, parent


# Generate the map with random shapes
map, labels = random_shapes((200, 300), min_shapes=5, max_shapes=20, num_channels=1)
map = (map == 255).astype(int) * 255  # Convert to binary (255 for free space, 0 for obstacles)

# Test the algorithm
start = (10, 10)
goal = (190, 290)
Dq = 10  # Step size
G, parent = grow_rrt_with_pixel_snapping(map, start, goal, Dq)

# Reconstruct the path from the parent dictionary
path = []
current = goal
while current is not None:
    path.append(current)
    current = parent[current]
path.reverse()

# Visualize the final path
plt.imshow(map, cmap='gray')
plt.plot(start[1], start[0], 'go', markersize=8)  # Start in green
plt.plot(goal[1], goal[0], 'bo', markersize=8)   # Goal in blue
for key in parent.keys():
    if parent[key] is not None:
        plt.plot([key[1], parent[key][1]], [key[0], parent[key][0]], 'ro-', linewidth=1, markersize=3)
for i in range(len(path) - 1):
    plt.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], 'y-', linewidth=2)  # Path in yellow
plt.show()

print("Path found:", path)

