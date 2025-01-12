import numpy as np
import matplotlib.pyplot as plt
import random

from skimage.draw import line_nd, random_shapes
# Initialize the empty map
#map = np.ones((200, 300)) * 255

from skimage.draw import line_nd, random_shapes
map, labels = random_shapes((200,300),20,5,num_channels=1)


# Define the start and goal nodes
qstart = (100, 150)
qgoal = (np.random.randint(0, map.shape[0]), np.random.randint(0, map.shape[1]))

# Initialize the graph
G = {qstart: []}  # Graph with nodes and neighbors
Dq = 10  # Step size for Δq

# Euclidean distance function
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Find the closest node in the graph to a random point
def find_closest_node(graph, qrand):
    min_dist = float("inf")
    closest_node = None
    for node in graph.keys():
        dist = euclidean_distance(node, qrand)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node

# Grow the RRT with controlled growth
def grow_rrt_with_controlled_growth(map, start, goal, Dq, iterations=500):
    """
    Grow an RRT tree with controlled growth by limiting steps to Δq.
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
        if random.random() < 0.1:  # 10% chance to bias toward the goal
            qrand = goal
        else:
            qrand = (np.random.randint(0, map.shape[0]), np.random.randint(0, map.shape[1]))

        # Find the closest node in the tree
        qnear = find_closest_node(G, qrand)

        # Compute the direction and step toward qrand
        theta = np.arctan2(qrand[0] - qnear[0], qrand[1] - qnear[1])
        qnew = (qnear[0] + Dq * np.sin(theta), qnear[1] + Dq * np.cos(theta))

        # Check if qnew is within bounds
        if 0 <= qnew[0] < map.shape[0] and 0 <= qnew[1] < map.shape[1]:
            # Add the new node to the tree
            G[qnear].append((qnew, euclidean_distance(qnear, qnew)))
            G[qnew] = []
            parent[qnew] = qnear

            # Visualize the edge in the tree
            plt.plot([qnear[1], qnew[1]], [qnear[0], qnew[0]], 'ro-', linewidth=2, markersize=3)
            plt.pause(0.01)  # Pause for animation

            # Check if qnew is close enough to the goal
            if euclidean_distance(qnew, goal) < Dq:
                G[qnew].append((goal, euclidean_distance(qnew, goal)))
                G[goal] = []
                parent[goal] = qnew
                print("Goal reached!")
                break

    plt.ioff()  # Turn off interactive mode
    plt.show()

    return G, parent

# Run the RRT algorithm with controlled growth
G, parent = grow_rrt_with_controlled_growth(map, qstart, qgoal, Dq)

# Reconstruct the path from the parent dictionary
path = []
current = qgoal
while current is not None:
    path.append(current)
    current = parent[current]
path.reverse()

# Visualize the path
plt.imshow(map, cmap='gray')
plt.plot(qstart[1], qstart[0], 'go', markersize=8)  # Start in green
plt.plot(qgoal[1], qgoal[0], 'bo', markersize=8)   # Goal in blue
for key in parent.keys():
    if parent[key] is not None:
        plt.plot([key[1], parent[key][1]], [key[0], parent[key][0]], 'ro-', linewidth=2, markersize=3)
for p in path:
    plt.plot(p[1], p[0], 'y.', markersize=5)  # Path in yellow
plt.show()

print("Path found:", path)

