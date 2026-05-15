import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
import os

# Define your skeleton edges (H36M format)
h36m_pts = [(3,2), (2,1), (1, 0), (0, 4), (4, 5), (5, 6),
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16),
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
zlabels = [f"{-tick:.1f}" for tick in zticks]
ax.set_zticklabels(zlabels)

# ---------------- CAMERA ----------------
R = np.array([[ 0.23675112, -0.00721419, -0.97154355],
              [-0.06633187,  0.99751916, -0.02357119],
              [ 0.96930335,  0.07002481,  0.23568525]])

t = np.array([0.81989035, -0.01071927, 0.57242022])

C1 = np.zeros(3)
C2 = -R.T @ t

ax.scatter(
    [-C1[2], -C2[2]],
    [ C1[0],  C2[0]],
    [-C1[1], -C2[1]],
    color='green',
    s=120
)

dir1 = np.array([0, 0, 1])
dir2 = R.T @ np.array([0, 0, 1])
cam_len = 0.25

ax.plot([-C1[2], -(C1[2] + cam_len * dir1[2])],
        [ C1[0],   C1[0] + cam_len * dir1[0]],
        [-C1[1], -(C1[1] + cam_len * dir1[1])],
        color='green')

ax.plot([-C2[2], -(C2[2] + cam_len * dir2[2])],
        [ C2[0],   C2[0] + cam_len * dir2[0]],
        [-C2[1], -(C2[1] + cam_len * dir2[1])],
        color='green')

# ---------------- SINGLE FRAME ----------------
FRAME_ID = 170  # <<< CHANGE THIS

def draw_frame(frame):
    x, y, z = keypoints_3d[frame, :, 0], keypoints_3d[frame, :, 1], keypoints_3d[frame, :, 2]

    ax.scatter(-z, x, -y, color='blue')

    for (i, j) in h36m_pts:
        ax.plot([-z[i], -z[j]],
                [x[i], x[j]],
                [-y[i], -y[j]],
                color='blue')

    if keypoints_3d_2 is not None:
        x2, y2, z2 = keypoints_3d_2[frame, :, 0], keypoints_3d_2[frame, :, 1], keypoints_3d_2[frame, :, 2]

        ax.scatter(-z2, x2, -y2, color='red')

        for (i, j) in h36m_pts:
            ax.plot([-z2[i], -z2[j]],
                    [x2[i], x2[j]],
                    [-y2[i], -y2[j]],
                    color='red')

# draw exactly one frame
draw_frame(FRAME_ID)

plt.show()