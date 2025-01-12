import matplotlib.pyplot as plt
from skimage.draw import line_nd, random_shapes

# Generate the map and labels
map, labels = random_shapes((200, 300), max_shapes=20, min_shapes=5, num_channels=1)

# Plot the map
plt.figure(figsize=(8, 6))
plt.imshow(map, cmap='gray')  # Use cmap='gray' for grayscale visualization
plt.title("Generated Random Shapes")
plt.axis('off')  # Turn off the axes for better visualization
plt.show()

for i in map:
  print(i)
