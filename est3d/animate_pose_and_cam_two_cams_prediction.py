import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# Define your skeleton edges (H36M format)
h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]

file_name_1 = "temp_seq/file_101_file_102_alternated.npy"
file_name_2 = None

keypoints_3d = np.load(file_name_1, allow_pickle=True)
keypoints_3d = np.asarray(keypoints_3d).astype(float)

if file_name_2:
    keypoints_3d_2 = np.load(file_name_2, allow_pickle=True)
    keypoints_3d_2 = np.asarray(keypoints_3d_2).astype(float)
else:
    keypoints_3d_2 = None

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Z')
ax.set_ylabel('X')
ax.set_zlabel('Y')

scale = 0.1
ax.set_xlim([-15 * scale, 0 * scale])
ax.set_ylim([-8 * scale, 7 * scale])
ax.set_zlim([-8 * scale, 7 * scale])

xticks = ax.get_xticks()
xlabels = [f"{-tick:.1f}" for tick in xticks]
ax.set_xticklabels(xlabels)

zticks = ax.get_zticks()
# Create labels as the negative of tick positions, formatted nicely
zlabels = [f"{-tick:.1f}" for tick in zticks]
# Set these labels on the x-axis
ax.set_zticklabels(zlabels)


R = np.array( [[ 0.23675112, -0.00721419, -0.97154355],
 [-0.06633187,  0.99751916, -0.02357119],
 [ 0.96930335,  0.07002481 , 0.23568525]])
t = np.array([ 0.81989035, -0.01071927 , 0.57242022])

C1 = np.zeros(3)
C2 = -R.T @ t

cam_scatter = ax.scatter(
    [-C1[2], -C2[2]],   # X-axis in your visualization
    [ C1[0],  C2[0]],   # Y-axis in your visualization
    [-C1[1], -C2[1]],   # Z-axis in your visualization
    color='green',
    s=120
)

dir1 = np.array([0, 0, 1])

# Camera 2 optical axis in world coords
dir2 = R.T @ np.array([0, 0, 1])

cam_len = 0.25

# Camera 1 direction
ax.plot(
    [-C1[2], -(C1[2] + cam_len * dir1[2])],
    [ C1[0],   C1[0] + cam_len * dir1[0]],
    [-C1[1], -(C1[1] + cam_len * dir1[1])],
    color='green',
    linewidth=2
)

# Camera 2 direction
ax.plot(
    [-C2[2], -(C2[2] + cam_len * dir2[2])],
    [ C2[0],   C2[0] + cam_len * dir2[0]],
    [-C2[1], -(C2[1] + cam_len * dir2[1])],
    color='green',
    linewidth=2
)


# Main skeleton
scatter = ax.scatter([], [], [], color='blue')
lines = [ax.plot([], [], [], color='blue')[0] for _ in h36m_pts]

# Second skeleton if provided
if keypoints_3d_2 is not None:
    scatter2 = ax.scatter([], [], [], color='red')
    lines2 = [ax.plot([], [], [], color='red')[0] for _ in h36m_pts]
else:
    scatter2, lines2 = None, []


keypoints_3d = keypoints_3d
def update(frame):
    x, y, z = keypoints_3d[frame, :, 0], keypoints_3d[frame, :, 1], keypoints_3d[frame, :, 2]
    scatter._offsets3d = (-z, x, -y)
    for line, (i, j) in zip(lines, h36m_pts):
        line.set_data([-z[i], -z[j]], [x[i], x[j]])
        line.set_3d_properties([-y[i], -y[j]])

    if keypoints_3d_2 is not None:
        x2, y2, z2 = keypoints_3d_2[frame, :, 0], keypoints_3d_2[frame, :, 1], keypoints_3d_2[frame, :, 2]
        scatter2._offsets3d = (-z2, x2, -y2)
        for line, (i, j) in zip(lines2, h36m_pts):
            line.set_data([-z2[i], -z2[j]], [x2[i], x2[j]])
            line.set_3d_properties([-y2[i], -y2[j]])

    return [scatter] + lines + ([scatter2] + lines2 if scatter2 else [])

base_1 = os.path.splitext(os.path.basename(file_name_1))[0]
if file_name_2:
    base_2 = os.path.splitext(os.path.basename(file_name_2))[0]
# Run animation
frame_count = min(len(keypoints_3d), len(keypoints_3d_2)) if keypoints_3d_2 is not None else len(keypoints_3d)
ani = FuncAnimation(fig, update, frames=frame_count, interval=8.33, blit=False)

#ani.save("animations/file_101_102_triangulated.gif", writer="ffmpeg", fps=30)

plt.show()