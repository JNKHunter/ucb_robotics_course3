import numpy as np
import matplotlib.pyplot as plt
import random
from heapq import heappush, heappop

# Initialize the empty map
map = np.ones((200, 300)) * 255

# Define the start and goal nodes
qstart = (100, 150)
qgoal = (np.random.randint(0, map.shape[0]), np.random.randint(0, map.shape[1]))

# Initialize the graph
G = {qstart: []}  # Graph with nodes and neighbors
parent = {qstart: None}  # Track parents for RRT
Dq = 10  # Step size for Δq

# Euclidean distance heuristic
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

# Grow the RRT tree
def grow_rrt(map, start, goal, iterations=500):
    for _ in range(iterations):
        # Randomly sample a configuration
        if random.random() < 0.1:  # 10% chance to bias towards the goal
            qrand = goal
        else:
            qrand = (np.random.randint(0, map.shape[0]), np.random.randint(0, map.shape[1]))

        # Find the closest node in the graph
        qnear = find_closest_node(G, qrand)

        # Grow the tree by Δq in the direction of qrand
        theta = np.arctan2(qrand[0] - qnear[0], qrand[1] - qnear[1])
        qnew = (qnear[0] + Dq * np.sin(theta), qnear[1] + Dq * np.cos(theta))

        # Check if qnew is within the bounds of the map
        if 0 <= qnew[0] < map.shape[0] and 0 <= qnew[1] < map.shape[1]:
            # Add the new node to the graph
            G[qnear].append((qnew, euclidean_distance(qnear, qnew)))  # Store edge with cost
            G[qnew] = []
            parent[qnew] = qnear

            # Stop if qnew is close enough to the goal
            if euclidean_distance(qnew, goal) < Dq:
                G[qnew].append((goal, euclidean_distance(qnew, goal)))
                G[goal] = []
                parent[goal] = qnew
                print("Goal reached!")
                break

    return G, parent

# A* Algorithm on the RRT graph
def astar_on_rrt(graph, start, goal):
    # Priority queue (min-heap) and cost dictionaries
    min_heap = []
    heappush(min_heap, (0 + euclidean_distance(start, goal), 0, start))  # (f(u), g(u), position)
    distances = {start: 0}  # Cost from start to each node
    predecessors = {start: None}

    while min_heap:
        f, g, current = heappop(min_heap)

        # Stop if we reach the goal
        if current == goal:
            print("A* reached the goal!")
            break

        # Explore neighbors
        for neighbor, cost in graph[current]:
            new_g = g + cost
            if neighbor not in distances or new_g < distances[neighbor]:
                distances[neighbor] = new_g
                f = new_g + euclidean_distance(neighbor, goal)
                heappush(min_heap, (f, new_g, neighbor))
                predecessors[neighbor] = current

    # Reconstruct the path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()

    return path

# Run the RRT algorithm to construct the tree
G, parent = grow_rrt(map, qstart, qgoal)

# Apply A* on the constructed RRT graph
path = astar_on_rrt(G, qstart, qgoal)

# Visualize the RRT and the path
plt.imshow(map, cmap='gray')
plt.plot(qstart[1], qstart[0], 'go', markersize=8)  # Start in green
plt.plot(qgoal[1], qgoal[0], 'bo', markersize=8)   # Goal in blue

# Draw the RRT tree
for node, neighbors in G.items():
    for neighbor, _ in neighbors:
        plt.plot([node[1], neighbor[1]], [node[0], neighbor[0]], 'ro-', linewidth=1, markersize=3)

# Draw the shortest path
for i in range(len(path) - 1):
    plt.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], 'y-', linewidth=2)  # Path in yellow

plt.show()

print("Shortest path using A* on RRT:", path)

