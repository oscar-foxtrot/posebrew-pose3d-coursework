import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- SKELETON ----------------
h36m_pts = [
    (3,2), (2,1), (1,0), (0,4), (4,5), (5,6),
    (13,12), (12,11), (11,8), (8,14), (14,15), (15,16),
    (8,9), (9,10), (8,7), (7,0)
]

file_name_1 = "temp_seq/file_101_file_102_alternated.npy"
file_name_2 = None

keypoints_3d = np.load(file_name_1, allow_pickle=True).astype(float)

if file_name_2:
    keypoints_3d_2 = np.load(file_name_2, allow_pickle=True).astype(float)
else:
    keypoints_3d_2 = None


# ---------------- SELECT FRAME ----------------
FRAME_ID = 310   # <<< CHANGE THIS

X = keypoints_3d[FRAME_ID]
if keypoints_3d_2 is not None:
    X2 = keypoints_3d_2[FRAME_ID]


# ---------------- FIGURE ----------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Z')
ax.set_ylabel('X')
ax.set_zlabel('Y')

scale = 0.07
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

# ---------------- CAMERA ----------------
C = np.array([0, 0, 0])

cam_len = 0.25
cam_dir = np.array([0.0, 0.0, 1.0])

ax.scatter(
    [-C[2]], [C[0]], [-C[1]],
    color='green', s=150
)

ax.plot(
    [-C[2], -(C[2] + cam_len * cam_dir[2])],
    [ C[0],   C[0] + cam_len * cam_dir[0]],
    [-C[1], -(C[1] + cam_len * cam_dir[1])],
    color='green',
    linewidth=3
)


# ---------------- MAIN SKELETON ----------------
ax.scatter(-X[:,2], X[:,0], -X[:,1], color='blue')

for i, j in h36m_pts:
    ax.plot(
        [-X[i,2], -X[j,2]],
        [ X[i,0],  X[j,0]],
        [-X[i,1], -X[j,1]],
        color='blue'
    )


# ---------------- SECOND SKELETON ----------------
if keypoints_3d_2 is not None:
    ax.scatter(-X2[:,2], X2[:,0], -X2[:,1], color='red')

    for i, j in h36m_pts:
        ax.plot(
            [-X2[i,2], -X2[j,2]],
            [ X2[i,0],  X2[j,0]],
            [-X2[i,1], -X2[j,1]],
            color='red'
        )


plt.show()