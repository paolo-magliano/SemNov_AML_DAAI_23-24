import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import imageio.v3 as iio

def load():
    f = open("/content/3D_OS_release_data/modelnet40_normal_resampled/airplane_0001.txt", "r")
    matrix = []
    for l in f:
        # use try-except block to protect the execution
        line = l.rstrip().split(',')
        line = line[:3]
        matrix.append(line)

    return matrix


# Create Figure:
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
figure = load()
ax.scatter3D(figure[:, 0], figure[:, 1], figure[:, 2])
# label the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Random Point Cloud")
# display:
plt.show()
