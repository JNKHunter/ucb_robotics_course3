import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from collections import defaultdict
import math

# Generate a random 20x30 grid map with 10% occupancy
#rows, cols = 20, 30
#grid_map = np.random.rand(rows, cols) < 0.1
grid_map = np.load('cspace.npy')
rows,cols = grid_map.shape
# Random start and goal points
start = (np.random.randint(0, rows), np.random.randint(0, cols))
goal = (np.random.randint(0, rows), np.random.randint(0, cols))

# Ensure start and goal are navigable
while grid_map[start] or grid_map[goal]:
    start = (np.random.randint(0, rows), np.random.randint(0, cols))
    goal = (np.random.randint(0, rows), np.random.randint(0, cols))

# Extended neighborhood for 8 directions
def getNeighbors(pos):
    directions = [
        (0, 1), (0, -1), (-1, 0), (1, 0),  # Cardinal directions
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal directions
    ]
    neighbors = []
    for di, dj in directions:
        ni, nj = pos[0] + di, pos[1] + dj
        if 0 <= ni < rows and 0 <= nj < cols and not grid_map[ni, nj]:  # Check bounds and navigability
            cost = math.sqrt(2) if abs(di) == 1 and abs(dj) == 1 else 1  # sqrt(2) for diagonals
            neighbors.append((cost, (ni, nj)))
    return neighbors

# Dijkstra's Algorithm
def dijkstra(grid_map, start, goal):
    # Priority queue (min-heap) and distances
    min_heap = []
    heappush(min_heap, (0, start))  # (cost to start, position)
    distances = defaultdict(lambda: float("inf"))
    distances[start] = 0
    predecessors = {start: None}  # To reconstruct the path
    visited = set()

    # Visualization setup
    plt.imshow(grid_map, cmap='gray')
    plt.ion()
    plt.plot(goal[1], goal[0], 'y*')  # Mark the goal
    plt.plot(start[1], start[0], 'b*')  # Mark the start

    while min_heap:
        current_cost, current = heappop(min_heap)

        if current in visited:
            continue
        visited.add(current)

        # Visualization: Show progress
        plt.plot(current[1], current[0], 'g.')
        plt.pause(0.000001)

        # Stop if we reach the goal
        if current == goal:
            print("Goal reached!")
            break

        for cost, neighbor in getNeighbors(current):
            new_cost = current_cost + cost
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                heappush(min_heap, (new_cost, neighbor))
                predecessors[neighbor] = current

    # Trace back the path
    path = []
    if goal in predecessors:  # If goal is reachable
        node = goal
        while node:
            path.append(node)
            node = predecessors[node]
        path.reverse()

        # Visualization: Color the shortest path in red
        for p in path:
            plt.plot(p[1], p[0], 'ro', markersize=5)
        #plt.pause(0.000001)
        plt.pause(1)
        plt.ioff()
        plt.show()
    else:
        print("No path found to the goal.")

    return path

# Run the Dijkstra algorithm
path = dijkstra(grid_map, start, goal)

# Output the result
if path:
    print("Shortest path found:", path)
else:
    print("No path found between", start, "and", goal)

